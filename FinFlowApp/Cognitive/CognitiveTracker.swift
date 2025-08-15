import Foundation
import HealthKit

final class CognitiveTracker {
    private let store = HKHealthStore()

    func fetchHRV(completion: @escaping (Result<Double, AppError>) -> Void) {
        #if USE_MOCK_SENSORS
        completion(.success(55 + Double.random(in: -8...8)))
        return
        #endif

        guard HKHealthStore.isHealthDataAvailable(),
              let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(.failure(.healthDataUnavailable)); return
        }
        let types: Set = [hrvType]
        store.requestAuthorization(toShare: nil, read: types) { [weak self] ok, _ in
            guard ok else { completion(.failure(.authorizationDenied)); return }
            self?.queryLatestHRV(type: hrvType, completion: completion)
        }
    }

    private func queryLatestHRV(type: HKQuantityType, completion: @escaping (Result<Double, AppError>) -> Void) {
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let q = HKSampleQuery(sampleType: type, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, samples, error in
            if let _ = error { completion(.failure(.healthDataUnavailable)); return }
            guard let sample = samples?.first as? HKQuantitySample else { completion(.failure(.healthDataUnavailable)); return }
            let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            completion(.success(ms))
        }
        store.execute(q)
    }
}