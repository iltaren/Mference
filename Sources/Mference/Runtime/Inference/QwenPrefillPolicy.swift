import Foundation

/// Batching choices for Qwen 3.6 prefill.
///
/// Routed experts run on the per-row kernel. With
/// `MFERENCE_PREFILL_GROUPED_EXPERTS=1` they run as grouped GEMM on chunk
/// tiles whose experts hold enough rows to fill the matrix tiles
/// (`PrefillGroupedExpertGate`), which reorders floating-point sums.
struct QwenPrefillPolicy: Equatable, Sendable {
    let groupedExperts: Bool

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        groupedExperts = PrefillGroupedExpertGate.optedIn(environment)
    }
}
