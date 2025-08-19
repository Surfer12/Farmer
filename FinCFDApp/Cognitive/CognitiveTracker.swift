import Foundation

#if os(iOS)
import HealthKit
final class CognitiveTracker {
	private let healthStore = HKHealthStore()

	func fetchHRV(completion: @escaping (Double?) -> Void) {
		guard HKHealthStore.isHealthDataAvailable() else { completion(nil); return }
		guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { completion(nil); return }
		let typesToShare: Set<HKSampleType> = []
		let typesToRead: Set<HKObjectType> = [hrvType]
		healthStore.requestAuthorization(toShare: typesToShare, read: typesToRead) { [weak self] granted, _ in
			guard granted else { completion(nil); return }
			self?.queryLatestHRV(type: hrvType, completion: completion)
		}
	}

	private func queryLatestHRV(type: HKQuantityType, completion: @escaping (Double?) -> Void) {
		let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
		let query = HKSampleQuery(sampleType: type, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, samples, _ in
			if let sample = samples?.first as? HKQuantitySample {
				let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
				completion(ms)
			} else {
				completion(nil)
			}
		}
		healthStore.execute(query)
	}
}
#else
final class CognitiveTracker {
	func fetchHRV(completion: @escaping (Double?) -> Void) {
		completion(nil)
	}
}
#endif