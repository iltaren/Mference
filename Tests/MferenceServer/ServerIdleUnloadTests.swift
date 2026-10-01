import Testing
@testable import MferenceServerCore

// MARK: - Arguments

@Suite("Server idle unload arguments")
struct ServerIdleUnloadArgumentTests {
    @Test func idleUnloadIsOffUnlessGiven() throws {
        #expect(try ServerArguments.parse(["--library"]).idleUnload == nil)
        #expect(try ServerArguments.parse(["--library", "--idle-unload", "off"]).idleUnload == nil)
    }

    @Test func secondsMinutesAndHoursAreAccepted() throws {
        let seconds: Duration? = try ServerArguments.parse(["--library", "--idle-unload", "30s"]).idleUnload
        let minutes: Duration? = try ServerArguments.parse(["--library", "--idle-unload", "10m"]).idleUnload
        let hours: Duration? = try ServerArguments.parse(["--library", "--idle-unload", "2h"]).idleUnload
        #expect(seconds == .seconds(30))
        #expect(minutes == .seconds(600))
        #expect(hours == .seconds(7200))
    }

    @Test(arguments: ["0s", "0m", "10", "10x", "-5m", "m", "", "1.5h", "10 m", "999999999999999999h"])
    func malformedDurationsAreRejected(value: String) {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--library", "--idle-unload", value])
        }
    }

    /// Single-model mode has no way to load its model again, so an idle
    /// timeout there is refused rather than silently ignored.
    @Test func aTimeoutNeedsTheLibrary() {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--idle-unload", "10m"])
        }
        #expect(throws: Never.self) {
            _ = try ServerArguments.parse(["--model", "m.gturbo", "--idle-unload", "off"])
        }
    }

    @Test func usageDocumentsIdleUnload() {
        #expect(ServerArguments.usage.contains("--idle-unload"))
    }
}

// MARK: - Coordinator idle timer

/// Stands in for `Task.sleep`, so the test decides when the idle timer wakes.
/// Each call is numbered from 0 in the order it arrives.
private actor IdleSleeper {
    private let honorsCancellation: Bool
    private(set) var requested: [Duration] = []
    private(set) var cancelled = 0
    private var pending: [Int: CheckedContinuation<Void, any Error>] = [:]
    private var arrivalWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    init(honorsCancellation: Bool = true) {
        self.honorsCancellation = honorsCancellation
    }

    func sleep(_ duration: Duration) async throws {
        let id = requested.count
        requested.append(duration)
        let ready = arrivalWaiters.filter { $0.count <= requested.count }
        arrivalWaiters.removeAll { $0.count <= requested.count }
        for waiter in ready { waiter.continuation.resume() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { pending[id] = $0 }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }

    /// Wakes sleep number `id` as if its time had elapsed.
    func fire(_ id: Int) {
        pending.removeValue(forKey: id)?.resume()
    }

    func waitForRequests(_ count: Int) async {
        guard requested.count < count else { return }
        await withCheckedContinuation { arrivalWaiters.append((count, $0)) }
    }

    private func cancel(_ id: Int) {
        cancelled += 1
        guard honorsCancellation else { return }
        pending.removeValue(forKey: id)?.resume(throwing: CancellationError())
    }
}

/// Counts idle-action runs; with a gate, each run parks until the test opens it.
private actor IdleRecorder {
    private(set) var started = 0
    private(set) var finished = 0
    private var gate: CheckedContinuation<Void, Never>?
    private let parks: Bool

    init(parks: Bool = false) {
        self.parks = parks
    }

    func run() async {
        started += 1
        if parks {
            await withCheckedContinuation { gate = $0 }
        }
        finished += 1
    }

    func open() {
        gate?.resume()
        gate = nil
    }
}

private actor TurnGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

private struct ConditionTimedOut: Error {}

/// Polls `condition` until it holds, failing after about two seconds.
private func eventually(_ condition: () async -> Bool) async throws {
    for _ in 0..<2_000 {
        if await condition() { return }
        try await Task.sleep(for: .milliseconds(1))
    }
    throw ConditionTimedOut()
}

@Suite("Server coordinator idle timer", .serialized)
struct ServerCoordinatorIdleTests {
    private static func coordinator(sleeper: IdleSleeper,
                                    idle: IdleRecorder,
                                    timeout: Duration? = .seconds(600)) -> ServerCoordinator {
        ServerCoordinator(queueLimit: 1,
                          idleTimeout: timeout,
                          onIdle: { await idle.run() },
                          sleep: { try await sleeper.sleep($0) })
    }

    @Test func idleActionRunsOnceAfterTheTimeout() async throws {
        let sleeper = IdleSleeper()
        let idle = IdleRecorder()
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle)
        await coordinator.startIdleTimer()
        await sleeper.waitForRequests(1)
        let requested: [Duration] = await sleeper.requested
        #expect(requested == [Duration.seconds(600)])

        await sleeper.fire(0)
        try await eventually { await idle.finished == 1 }
        try await eventually { await !coordinator.isActive }
        // Nothing is armed again until a request has run.
        #expect(await !coordinator.isIdleTimerArmed)
        #expect(await idle.started == 1)
    }

    @Test func aRequestCancelsThePendingTimerAndRearmsItWhenItFinishes() async throws {
        let sleeper = IdleSleeper()
        let idle = IdleRecorder()
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle)
        await coordinator.startIdleTimer()
        await sleeper.waitForRequests(1)

        #expect(try await coordinator.run { 7 } == 7)
        try await eventually { await sleeper.cancelled == 1 }
        await sleeper.waitForRequests(2)
        #expect(await idle.started == 0)

        await sleeper.fire(1)
        try await eventually { await idle.finished == 1 }
    }

    /// A timer that wakes after a turn has begun — its sleep ignored the
    /// cancellation — must not run the idle action under that turn.
    @Test func aTimerThatWakesDuringATurnDoesNothing() async throws {
        let sleeper = IdleSleeper(honorsCancellation: false)
        let idle = IdleRecorder()
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle)
        await coordinator.startIdleTimer()
        await sleeper.waitForRequests(1)

        let gate = TurnGate()
        let request = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        try await eventually { await coordinator.isActive }
        await sleeper.fire(0)
        try await Task.sleep(for: .milliseconds(50))
        #expect(await idle.started == 0)

        await gate.open()
        #expect(try await request.value == 1)
        await sleeper.waitForRequests(2)
        await sleeper.fire(1)
        try await eventually { await idle.finished == 1 }
        #expect(await idle.started == 1)
    }

    /// The idle action holds the turn, so a request that arrives meanwhile
    /// queues behind it instead of racing it.
    @Test func aRequestArrivingDuringTheIdleActionWaitsForIt() async throws {
        let sleeper = IdleSleeper()
        let idle = IdleRecorder(parks: true)
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle)
        await coordinator.startIdleTimer()
        await sleeper.waitForRequests(1)
        await sleeper.fire(0)
        try await eventually { await idle.started == 1 }

        let request = Task { try await coordinator.run { 5 } }
        try await eventually { await coordinator.queuedCount == 1 }
        #expect(await idle.finished == 0)

        await idle.open()
        #expect(try await request.value == 5)
        #expect(await idle.finished == 1)
    }

    @Test func noTimeoutMeansNoTimer() async throws {
        let sleeper = IdleSleeper()
        let idle = IdleRecorder()
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle, timeout: nil)
        await coordinator.startIdleTimer()
        #expect(try await coordinator.run { 1 } == 1)
        #expect(await !coordinator.isIdleTimerArmed)
        #expect(await sleeper.requested.isEmpty)
    }

    @Test func shutdownCancelsThePendingTimer() async throws {
        let sleeper = IdleSleeper()
        let idle = IdleRecorder()
        let coordinator = Self.coordinator(sleeper: sleeper, idle: idle)
        await coordinator.startIdleTimer()
        await sleeper.waitForRequests(1)

        await coordinator.shutdown()
        try await eventually { await sleeper.cancelled == 1 }
        #expect(await !coordinator.isIdleTimerArmed)
        #expect(await idle.started == 0)
    }
}
