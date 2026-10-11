use crate::error::{Ml5Error, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

#[derive(Debug, Clone)]
enum Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    UintArray(Vec<u64>),
    Skipped,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GgufInfo {
    pub architecture: String,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Vec<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub sliding_window: Option<u64>,
    pub pooling_type: Option<u64>,
    #[serde(skip)]
    pub chat_template: Option<String>,
    pub vision_encoder: bool,
}

const WANTED_SUFFIXES: &[&str] = &[
    ".context_length",
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".attention.sliding_window",
    ".pooling_type",
];

const WANTED_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    "general.size_label",
    "general.file_type",
    "tokenizer.chat_template",
    "clip.has_vision_encoder",
];

const MAX_STRING: u64 = 64 * 1024 * 1024;
const MAX_ARRAY: u64 = 1 << 32;

struct Reader<R: Read + Seek> {
    inner: BufReader<R>,
}

impl<R: Read + Seek> Reader<R> {
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        let n = i64::try_from(n).map_err(|_| bad("skip length overflow"))?;
        self.inner.seek_relative(n)?;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(bad("string too long"));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    fn skip_string(&mut self) -> Result<()> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(bad("string too long"));
        }
        self.skip(len)
    }
    fn scalar(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::Uint(self.u8()? as u64),
            1 => Value::Int(self.u8()? as i8 as i64),
            2 => {
                let mut b = [0u8; 2];
                self.inner.read_exact(&mut b)?;
                Value::Uint(u16::from_le_bytes(b) as u64)
            }
            3 => {
                let mut b = [0u8; 2];
                self.inner.read_exact(&mut b)?;
                Value::Int(i16::from_le_bytes(b) as i64)
            }
            4 => Value::Uint(self.u32()? as u64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            10 => Value::Uint(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            other => return Err(bad(&format!("unknown value type {other}"))),
        })
    }
    fn scalar_size(ty: u32) -> Option<u64> {
        match ty {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        }
    }
    fn value(&mut self, ty: u32, keep: bool) -> Result<Value> {
        if ty == 9 {
            let inner = self.u32()?;
            let n = self.u64()?;
            if n > MAX_ARRAY {
                return Err(bad("array too long"));
            }
            if keep && matches!(inner, 0 | 2 | 4 | 5 | 10 | 11) && n <= 4096 {
                let mut out = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    out.push(match self.scalar(inner)? {
                        Value::Uint(v) => v,
                        Value::Int(v) => v.max(0) as u64,
                        _ => 0,
                    });
                }
                return Ok(Value::UintArray(out));
            }
            if let Some(size) = Self::scalar_size(inner) {
                self.skip(size.checked_mul(n).ok_or_else(|| bad("array size overflow"))?)?;
            } else if inner == 8 {
                for _ in 0..n {
                    self.skip_string()?;
                }
            } else if inner == 9 {
                for _ in 0..n {
                    self.value(9, false)?;
                }
            } else {
                return Err(bad(&format!("unknown array element type {inner}")));
            }
            return Ok(Value::Skipped);
        }
        if !keep && ty == 8 {
            self.skip_string()?;
            return Ok(Value::Skipped);
        }
        self.scalar(ty)
    }
}

fn bad(msg: &str) -> Ml5Error {
    Ml5Error::InvalidRequest(format!("not a readable GGUF file: {msg}"))
}

fn wanted(key: &str) -> bool {
    WANTED_KEYS.contains(&key) || WANTED_SUFFIXES.iter().any(|s| key.ends_with(s))
}

pub fn read_info(path: &Path) -> Result<GgufInfo> {
    let file = std::fs::File::open(path)?;
    let mut r = Reader {
        inner: BufReader::with_capacity(256 * 1024, file),
    };
    let mut magic = [0u8; 4];
    r.inner.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(bad("missing GGUF magic"));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(bad(&format!("unsupported GGUF version {version}")));
    }
    let _tensors = r.u64()?;
    let kv_count = r.u64()?;
    let mut kv: HashMap<String, Value> = HashMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let keep = wanted(&key);
        let value = r.value(ty, keep)?;
        if keep {
            kv.insert(key, value);
        }
    }

    let arch = match kv.get("general.architecture") {
        Some(Value::Str(s)) => s.clone(),
        _ => String::new(),
    };
    let num = |k: &str| -> Option<u64> {
        match kv.get(&format!("{arch}.{k}")) {
            Some(Value::Uint(v)) => Some(*v),
            Some(Value::Int(v)) if *v >= 0 => Some(*v as u64),
            Some(Value::Float(v)) if *v >= 0.0 => Some(*v as u64),
            Some(Value::UintArray(a)) => a.iter().copied().max(),
            _ => None,
        }
    };
    let text = |k: &str| -> Option<String> {
        match kv.get(k) {
            Some(Value::Str(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    };
    let head_count_kv = match kv.get(&format!("{arch}.attention.head_count_kv")) {
        Some(Value::UintArray(a)) => a.clone(),
        Some(Value::Uint(v)) => vec![*v],
        Some(Value::Int(v)) if *v >= 0 => vec![*v as u64],
        _ => Vec::new(),
    };
    Ok(GgufInfo {
        name: text("general.name"),
        size_label: text("general.size_label"),
        file_type: match kv.get("general.file_type") {
            Some(Value::Uint(v)) => Some(*v as u32),
            Some(Value::Int(v)) if *v >= 0 => Some(*v as u32),
            _ => None,
        },
        context_length: num("context_length"),
        block_count: num("block_count"),
        embedding_length: num("embedding_length"),
        head_count: num("attention.head_count"),
        head_count_kv,
        key_length: num("attention.key_length"),
        value_length: num("attention.value_length"),
        sliding_window: num("attention.sliding_window"),
        pooling_type: num("pooling_type"),
        chat_template: text("tokenizer.chat_template"),
        vision_encoder: matches!(kv.get("clip.has_vision_encoder"), Some(Value::Bool(true))),
        architecture: arch,
    })
}

impl GgufInfo {
    pub fn quantization(&self) -> &'static str {
        match self.file_type {
            Some(0) => "F32",
            Some(1) => "F16",
            Some(2) => "Q4_0",
            Some(3) => "Q4_1",
            Some(7) => "Q8_0",
            Some(8) => "Q5_0",
            Some(9) => "Q5_1",
            Some(10) => "Q2_K",
            Some(11) => "Q3_K_S",
            Some(12) => "Q3_K_M",
            Some(13) => "Q3_K_L",
            Some(14) => "Q4_K_S",
            Some(15) => "Q4_K_M",
            Some(16) => "Q5_K_S",
            Some(17) => "Q5_K_M",
            Some(18) => "Q6_K",
            Some(19) => "IQ2_XXS",
            Some(20) => "IQ2_XS",
            Some(21) => "Q2_K_S",
            Some(22) => "IQ3_XS",
            Some(23) => "IQ3_XXS",
            Some(24) => "IQ1_S",
            Some(25) => "IQ4_NL",
            Some(26) => "IQ3_S",
            Some(27) => "IQ3_M",
            Some(28) => "IQ2_S",
            Some(29) => "IQ2_M",
            Some(30) => "IQ4_XS",
            Some(31) => "IQ1_M",
            Some(32) => "BF16",
            Some(36) => "TQ1_0",
            Some(37) => "TQ2_0",
            Some(38) => "MXFP4",
            _ => "unknown",
        }
    }

    pub fn supports_tools(&self) -> bool {
        self.chat_template
            .as_deref()
            .is_some_and(|t| t.contains("tools") || t.contains("tool_call"))
    }

    pub fn supports_thinking(&self) -> bool {
        self.chat_template.as_deref().is_some_and(|t| {
            t.contains("<think>")
                || t.contains("enable_thinking")
                || t.contains("reasoning_content")
                || t.contains("<|channel|>")
                || t.contains("thinking")
        })
    }

    pub fn is_embedding_model(&self) -> bool {
        const ENCODERS: &[&str] = &[
            "bert",
            "nomic-bert",
            "nomic-bert-moe",
            "jina-bert-v2",
            "jina-bert-v3",
            "neo-bert",
            "modern-bert",
            "t5encoder",
            "xlm-roberta",
            "eurobert",
        ];
        ENCODERS.contains(&self.architecture.as_str()) || self.pooling_type.is_some_and(|p| p > 0)
    }

    pub fn is_recurrent(&self) -> bool {
        let a = self.architecture.as_str();
        a.contains("mamba") || a.contains("rwkv")
    }

    pub fn kv_bytes_per_token(&self, cache_bytes_per_element: f64) -> u64 {
        if self.is_recurrent() {
            return 0;
        }
        let layers = self.block_count.unwrap_or(32);
        let embd = self.embedding_length.unwrap_or(4096);
        let heads = self.head_count.unwrap_or(32).max(1);
        let head_dim = embd / heads;
        let k = self.key_length.unwrap_or(head_dim);
        let v = self.value_length.unwrap_or(head_dim);
        let total_kv_heads: u64 = match self.head_count_kv.len() {
            0 => heads * layers,
            1 => self.head_count_kv[0] * layers,
            n if n as u64 == layers => self.head_count_kv.iter().sum(),
            _ => self.head_count_kv.iter().copied().max().unwrap_or(heads) * layers,
        };
        ((total_kv_heads * (k + v)) as f64 * cache_bytes_per_element).ceil() as u64
    }

    pub fn parameter_label(&self, file_size: u64) -> String {
        if let Some(label) = &self.size_label {
            return label.clone();
        }
        let bits = match self.quantization() {
            "F32" => 32.0,
            "F16" | "BF16" => 16.0,
            "Q8_0" => 8.5,
            "Q6_K" => 6.6,
            "Q5_0" | "Q5_1" | "Q5_K_S" | "Q5_K_M" => 5.5,
            "Q4_0" | "Q4_1" | "Q4_K_S" | "Q4_K_M" | "IQ4_NL" | "IQ4_XS" | "MXFP4" => 4.6,
            "Q3_K_S" | "Q3_K_M" | "Q3_K_L" | "IQ3_XS" | "IQ3_XXS" | "IQ3_S" | "IQ3_M" => 3.6,
            _ => 2.8,
        };
        let params = file_size as f64 * 8.0 / bits;
        if params >= 1e9 {
            format!("{:.1}B", params / 1e9)
        } else {
            format!("{:.0}M", params / 1e6)
        }
    }
}

pub fn cache_type_bytes(kind: &str) -> f64 {
    match kind {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 1.0625,
        "q5_0" | "q5_1" => 0.6875,
        "q4_0" | "q4_1" | "iq4_nl" => 0.5625,
        _ => 2.0,
    }
}
