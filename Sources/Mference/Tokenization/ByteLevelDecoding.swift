import Foundation
import Hub

/// Incremental decode for a GPT-2-style byte-level BPE (Qwen) whose
/// `tokenizer_config.json` turns clean-up off.
///
/// The library decodes such a tokenizer by dropping special IDs, mapping each
/// remaining token to its string, joining runs of ordinary tokens, mapping
/// every character back to its byte and decoding each run once with
/// `String(decoding:as: UTF8.self)`; an added token ends the run and is kept
/// verbatim. Re-running that over the whole reply for every new token costs
/// O(n) per token. This reproduces it one token at a time: bytes accumulate,
/// and only a trailing scalar that later bytes could still complete is held
/// back, so the streamed text equals the library decode exactly.
struct ByteLevelDecoding: Sendable {
    /// `added_tokens[special == true]`: removed by `skipSpecialTokens`.
    let specialTokenIDs: Set<Int32>
    /// Every added token's content: kept verbatim, and it ends a byte run.
    let addedTokens: Set<String>

    /// Non-nil only for the declaration this type reproduces: a plain
    /// `ByteLevel` decoder and `clean_up_tokenization_spaces: false`.
    init?(tokenizerData: Config, cleanUpTokenizationSpaces: Bool) {
        guard !cleanUpTokenizationSpaces,
              tokenizerData["decoder"].type.string() == "ByteLevel" else { return nil }
        var specials: Set<Int32> = []
        var added: Set<String> = []
        for token in tokenizerData["addedTokens"].array(or: []) {
            guard let id = token["id"].integer(), let content = token.content.string() else { continue }
            added.insert(content)
            if token["special"].boolean(or: false), let value = Int32(exactly: id) {
                specials.insert(value)
            }
        }
        self.specialTokenIDs = specials
        self.addedTokens = added
    }

    /// GPT-2 `bytes_to_unicode`, inverted: scalar value -> byte. The printable
    /// bytes map to themselves and the rest to U+0100 onwards, so every
    /// scalar the alphabet uses is below U+0144.
    static let byteForScalar: [UInt16] = {
        var table = [UInt16](repeating: .max, count: 0x144)
        var next = 0x100
        for byte in 0..<256 {
            let printable = (0x21...0x7E).contains(byte) || (0xA1...0xAC).contains(byte)
                || (0xAE...0xFF).contains(byte)
            if printable {
                table[byte] = UInt16(byte)
            } else {
                table[next] = UInt16(byte)
                next += 1
            }
        }
        return table
    }()
}

/// Bytes of the current run that do not yet form complete scalars.
struct ByteLevelRun {
    private var pending: [UInt8] = []

    /// Appends one ordinary token and returns the text it completes.
    mutating func push(_ token: String) -> String {
        for scalar in token.unicodeScalars {
            let value = Int(scalar.value)
            if value < ByteLevelDecoding.byteForScalar.count,
               case let byte = ByteLevelDecoding.byteForScalar[value], byte != .max {
                pending.append(UInt8(byte))
            } else {
                // Outside the byte alphabet (the library traps here); keep
                // the character itself.
                pending.append(contentsOf: String(scalar).utf8)
            }
        }
        let held = Self.incompleteTailLength(pending)
        let ready = pending.count - held
        guard ready > 0 else { return "" }
        let text = String(decoding: pending[..<ready], as: UTF8.self)
        pending.removeFirst(ready)
        return text
    }

    /// Ends the run as the library does at an added token or the end: an
    /// incomplete scalar decodes to U+FFFD.
    mutating func commit() -> String {
        guard !pending.isEmpty else { return "" }
        defer { pending.removeAll(keepingCapacity: true) }
        return String(decoding: pending, as: UTF8.self)
    }

    /// Length of a trailing lead byte plus continuation bytes that a later
    /// byte could still complete into one scalar (Unicode table 3-7). Such a
    /// tail starts at a lead byte, which no earlier sequence can absorb, so
    /// decoding before it and after it separately equals decoding the whole.
    static func incompleteTailLength(_ bytes: [UInt8]) -> Int {
        let n = bytes.count
        var start = n - 1
        while start >= 0, start >= n - 3, bytes[start] & 0xC0 == 0x80 { start -= 1 }
        guard start >= 0, start >= n - 3 else { return 0 }
        let lead = bytes[start]
        let need: Int
        switch lead {
        case 0xC2...0xDF: need = 2
        case 0xE0...0xEF: need = 3
        case 0xF0...0xF4: need = 4
        default: return 0
        }
        let have = n - start
        guard have < need else { return 0 }
        if have >= 2 {
            let second = bytes[start + 1]
            let range: ClosedRange<UInt8> = switch lead {
            case 0xE0: 0xA0...0xBF
            case 0xED: 0x80...0x9F
            case 0xF0: 0x90...0xBF
            case 0xF4: 0x80...0x8F
            default: 0x80...0xBF
            }
            guard range.contains(second) else { return 0 }
        }
        return have
    }
}
