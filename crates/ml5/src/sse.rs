use anyhow::{bail, Result};

#[derive(Default)]
pub struct Decoder {
    pending: Vec<u8>,
    data: Vec<String>,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut consumed = 0;
        while let Some(end) = self.pending[consumed..].iter().position(|b| *b == b'\n') {
            let end = consumed + end;
            let line = std::str::from_utf8(&self.pending[consumed..end])?.trim_end_matches('\r');
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(self.data.join("\n"));
                    self.data.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data
                    .push(data.strip_prefix(' ').unwrap_or(data).to_string());
            }
            consumed = end + 1;
        }
        self.pending.drain(..consumed);
        if self.pending.len() + self.data.iter().map(String::len).sum::<usize>() > 4 * 1024 * 1024 {
            bail!("Server sent an oversized SSE event");
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_boundary_preserves_unicode_and_crlf() {
        let input = ": heartbeat\r\ndata: {\"text\":\"日本語 🦀\"}\r\n\r\ndata: one\ndata: two\n\n";
        for split in 0..=input.len() {
            let mut decoder = Decoder::default();
            let mut output = decoder.push(&input.as_bytes()[..split]).unwrap();
            output.extend(decoder.push(&input.as_bytes()[split..]).unwrap());
            assert_eq!(output, vec!["{\"text\":\"日本語 🦀\"}", "one\ntwo"]);
        }
    }
}
