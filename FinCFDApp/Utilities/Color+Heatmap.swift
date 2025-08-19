import Foundation
#if canImport(UIKit)
import UIKit
public typealias PlatformColor = UIColor
#else
import AppKit
public typealias PlatformColor = NSColor
#endif

enum HeatmapColorMapper {
	static func color(for value: Double) -> PlatformColor {
		let clamped = max(0.0, min(1.0, value))
		let r = CGFloat(clamped)
		let g = CGFloat(0.0)
		let b = CGFloat(1.0 - clamped)
		return PlatformColor(red: r, green: g, blue: b, alpha: 1.0)
	}
}