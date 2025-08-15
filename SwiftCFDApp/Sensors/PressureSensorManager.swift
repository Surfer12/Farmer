import Foundation
import Combine
import CoreBluetooth

final class PressureSensorManager: NSObject, ObservableObject {
    @Published var pressures: [Float] = []

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var characteristic: CBCharacteristic?
    private var timer: Timer?

    func startScanning() {
        #if os(iOS)
        central = CBCentralManager(delegate: self, queue: .main)
        #else
        // On macOS or if Bluetooth is unavailable, start mock
        startMock()
        #endif
    }

    func stop() {
        central?.stopScan()
        timer?.invalidate()
        timer = nil
    }

    private func startMock() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            // Simulate ~30% pressure differential pattern
            let base: Float = .random(in: 0.3...0.6)
            let delta: Float = 0.3
            let values = (0..<128).map { i -> Float in
                let phase = sin(Float(i) / 12.0)
                return max(0, min(1, base + delta * 0.5 * Float(phase)))
            }
            self.pressures = values
        }
    }
}

extension PressureSensorManager: CBCentralManagerDelegate, CBPeripheralDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            central.scanForPeripherals(withServices: [BluetoothConfig.pressureService])
        case .unsupported, .unauthorized, .poweredOff, .resetting, .unknown:
            startMock()
        @unknown default:
            startMock()
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        self.peripheral = peripheral
        central.stopScan()
        peripheral.delegate = self
        central.connect(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.discoverServices([BluetoothConfig.pressureService])
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        if let service = peripheral.services?.first(where: { $0.uuid == BluetoothConfig.pressureService }) {
            peripheral.discoverCharacteristics([BluetoothConfig.pressureCharacteristic], for: service)
        } else {
            startMock()
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let c = service.characteristics?.first(where: { $0.uuid == BluetoothConfig.pressureCharacteristic }) else {
            startMock(); return
        }
        characteristic = c
        peripheral.setNotifyValue(true, for: c)
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil, let data = characteristic.value else { return }
        // Expect an array of Float32 pressures [0,1]
        let count = data.count / MemoryLayout<Float32>.size
        var arr = [Float32](repeating: 0, count: count)
        _ = arr.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        pressures = arr.map { max(0, min(1, Float($0))) }
    }
}