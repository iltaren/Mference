import Foundation

/// Batching choices for Qwen 3.6 prefill.
///
/// Routed experts run as grouped GEMM on chunk tiles whose experts hold enough
/// rows to fill the matrix tiles (`PrefillGroupedExpertGate`). That reorders
/// floating-point sums against the per-row kernel;
/// `MFERENCE_QWEN_PREFILL_ROW_EXPERTS=1` keeps the per-row kernel everywhere.
struct QwenPrefillPolicy: Equatable, Sendable {
    let groupedExperts: Bool

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        groupedExperts = environment["MFERENCE_QWEN_PREFILL_ROW_EXPERTS"] != "1"
    }
}
