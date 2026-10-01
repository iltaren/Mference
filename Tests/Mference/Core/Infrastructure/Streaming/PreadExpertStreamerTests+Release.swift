import Darwin
import Foundation
import Metal
import Testing

@testable import Mference

extension PreadExpertStreamerTests {
  /// The library server releases a model in a long-lived process, so the slot
  /// slab must go back to the system when the streamer does, not into an
  /// allocator cache that keeps its pages dirty and counted against the server.
  @Test func releasingTheStreamerReturnsTheSlabToTheSystem() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    // 2,048 slots of two pages each: a 64 MiB slab with 16 KiB pages.
    let slotCount = 2_048
    let slabBytes = slotCount * Self.expertStride
    let before = try Self.physicalFootprint()
    try autoreleasepool {
      let streamer = try PreadExpertStreamer(
        layout: Self.makeLayout(path: url.path), device: device, slotCount: slotCount)
      memset(streamer.slotMapBinding.slab.contents(), 0x5A, slabBytes)
      let resident = try Self.physicalFootprint() - before
      #expect(resident >= slabBytes * 9 / 10, "slab pages were not touched: \(resident) bytes")
    }
    var retained = try Self.physicalFootprint() - before
    for _ in 0..<100 where retained > slabBytes / 10 {
      usleep(10_000)
      retained = try Self.physicalFootprint() - before
    }
    #expect(retained <= slabBytes / 10, "\(retained) bytes of a \(slabBytes)-byte slab stayed resident")
  }

  @Test func slotSlabKeepsItsAlignment() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 3)
    let base = UInt(bitPattern: streamer.slotMapBinding.slab.contents())
    #expect(base % UInt(PreadExpertStreamer.scratchAlignment) == 0)
    #expect(streamer.slotMapBinding.slab.length == 3 * Self.expertStride)
  }

  struct FootprintUnavailable: Error {}

  static func physicalFootprint() throws -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                       / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { throw FootprintUnavailable() }
    return Int(info.phys_footprint)
  }
}
