//! Tokenization for context management.
//!
//! This module provides tokenization functionality to manage LLM context windows
//! and ensure we don't exceed token limits.

/// Trait for tokenization.
pub trait Tokenizer: Send + Sync {
    /// Encode text into tokens.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode tokens back to text.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Count tokens in text (convenience method).
    fn count_tokens(&self, text: &str) -> usize {
        self.encode(text).len()
    }
}

/// Check if a character is a CJK (Chinese/Japanese/Korean) character.
///
/// CJK characters occupy more token budget than Latin characters because
/// each character often maps to 1-2 tokens in subword tokenizers (e.g., tiktoken).
pub fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{1100}'..='\u{11FF}' |  // Hangul Jamo
        '\u{3040}'..='\u{309F}' |  // Hiragana
        '\u{30A0}'..='\u{30FF}' |  // Katakana
        '\u{3130}'..='\u{318F}' |  // Hangul Compatibility Jamo
        '\u{3400}'..='\u{4DBF}' |  // CJK Unified Ideographs Extension A
        '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
        '\u{A960}'..='\u{A97F}' |  // Hangul Jamo Extended-A
        '\u{AC00}'..='\u{D7AF}' |  // Hangul Syllables (Korean)
        '\u{D7B0}'..='\u{D7FF}' |  // Hangul Jamo Extended-B
        '\u{F900}'..='\u{FAFF}' |  // CJK Compatibility Ideographs
        '\u{20000}'..='\u{2A6DF}'  // CJK Unified Ideographs Extension B
    )
}

/// Estimate token count with language awareness.
///
/// - CJK characters: ~2 tokens each (1 char = 1-2 tokens in LLM tokenizers)
/// - Latin/ASCII text: ~4 characters per token
pub fn estimate_tokens_for_text(text: &str) -> usize {
    let mut cjk_tokens = 0usize;
    let mut latin_bytes = 0usize;

    for c in text.chars() {
        if is_cjk_char(c) {
            cjk_tokens += 2; // Each CJK char ≈ 2 tokens
        } else {
            latin_bytes += c.len_utf8();
        }
    }

    let latin_tokens = (latin_bytes as f32 / 4.0).ceil() as usize;
    cjk_tokens + latin_tokens
}

/// Simple tokenizer that estimates tokens (for testing and fallback).
/// Uses a language-aware heuristic:
/// - CJK (Korean/Chinese/Japanese): ~2 tokens per character
/// - Latin/ASCII: ~4 characters per token
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    /// Create a new simple tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let estimated_tokens = estimate_tokens_for_text(text);
        (0..estimated_tokens).map(|i| i as u32).collect()
    }

    fn decode(&self, _tokens: &[u32]) -> String {
        // Simple tokenizer doesn't support actual decoding
        String::from("[decoded text]")
    }

    fn count_tokens(&self, text: &str) -> usize {
        // Language-aware heuristic:
        // - CJK (Korean/Chinese/Japanese): ~2 tokens per character (each char is its own token)
        // - Latin/ASCII/other: ~4 characters per token (GPT average)
        // Also count words as minimum for Latin text
        let token_estimate = estimate_tokens_for_text(text);
        let word_count = text.split_whitespace().count();
        token_estimate.max(word_count)
    }
}

/// Mock tokenizer for testing with configurable token counts.
pub struct MockTokenizer {
    tokens_per_char: f32,
}

impl MockTokenizer {
    /// Create a new mock tokenizer.
    pub fn new() -> Self {
        Self {
            tokens_per_char: 0.25, // Default: 4 chars per token
        }
    }

    /// Create with custom token rate.
    pub fn with_rate(tokens_per_char: f32) -> Self {
        Self { tokens_per_char }
    }
}

impl Default for MockTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let token_count = (text.len() as f32 * self.tokens_per_char).ceil() as usize;
        (0..token_count).map(|i| i as u32).collect()
    }

    fn decode(&self, _tokens: &[u32]) -> String {
        String::from("[mock decoded]")
    }

    fn count_tokens(&self, text: &str) -> usize {
        (text.len() as f32 * self.tokens_per_char).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_count() {
        let tokenizer = SimpleTokenizer::new();

        // Short text
        let count = tokenizer.count_tokens("Hello world");
        assert!(count > 0);
        assert!(count < 10);

        // Longer text
        let long_text = "This is a much longer piece of text that should have more tokens";
        let long_count = tokenizer.count_tokens(long_text);
        assert!(long_count > count);
    }

    #[test]
    fn test_simple_tokenizer_encode_decode() {
        let tokenizer = SimpleTokenizer::new();

        let tokens = tokenizer.encode("test");
        assert!(tokens.len() > 0);

        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_mock_tokenizer_custom_rate() {
        let tokenizer = MockTokenizer::with_rate(0.5); // 2 chars per token

        let count = tokenizer.count_tokens("test"); // 4 chars = 2 tokens
        assert_eq!(count, 2);
    }

    #[test]
    fn test_mock_tokenizer_default() {
        let tokenizer = MockTokenizer::default();

        // 4 chars per token by default
        let count = tokenizer.count_tokens("test"); // 4 chars = 1 token
        assert_eq!(count, 1);
    }

    #[test]
    fn test_tokenizer_trait() {
        fn test_tokenizer<T: Tokenizer>(tokenizer: &T) {
            let count = tokenizer.count_tokens("hello");
            assert!(count > 0);
        }

        test_tokenizer(&SimpleTokenizer::new());
        test_tokenizer(&MockTokenizer::new());
    }
}
