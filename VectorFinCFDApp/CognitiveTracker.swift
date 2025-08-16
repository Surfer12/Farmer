// SPDX-License-Identifier: GPL-3.0-only
import Foundation
import HealthKit

final class CognitiveTracker {
	private let healthStore = HKHealthStore()
	
	enum HealthError: Error {
		case unavailable
		case queryFailed
	}
	
	func fetchHRV(completion: @escaping (Result<Double, Error>) -> Void) {
		guard HKHealthStore.isHealthDataAvailable(),
				let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
			completion(.failure(HealthError.unavailable))
			return
		}
		
		healthStore.requestAuthorization(toShare: [], read: [hrvType]) { [weak self] granted, error in
			guard granted, error == nil, let self else {
				completion(.failure(error ?? HealthError.unavailable))
				return
			}
			let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
			let query = HKSampleQuery(sampleType: hrvType, predicate: nil, limit: 1, sortDescriptors: [sort]) { _, samples, err in
				if let err = err {
					completion(.failure(err))
					return
				}
				guard let sample = samples?.first as? HKQuantitySample else {
					completion(.failure(HealthError.queryFailed))
					return
				}
				let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
				completion(.success(ms))
			}
			self.healthStore.execute(query)
		}
	}
}