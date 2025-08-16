import Foundation
import SceneKit

#if os(iOS) || os(tvOS)
import UIKit
public typealias PlatformColor = UIColor
public typealias PlatformImage = UIImage
public typealias PlatformBezierPath = UIBezierPath
#elseif os(macOS)
import AppKit
public typealias PlatformColor = NSColor
public typealias PlatformImage = NSImage
public typealias PlatformBezierPath = NSBezierPath
#endif

extension PlatformColor {
    static func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> PlatformColor {
        #if os(macOS)
        return PlatformColor(calibratedRed: r, green: g, blue: b, alpha: a)
        #else
        return PlatformColor(red: r, green: g, blue: b, alpha: a)
        #endif
    }
}

#if os(macOS)
extension NSImage {
    convenience init(cgImage: CGImage, width: Int, height: Int) {
        self.init(cgImage: cgImage, size: NSSize(width: width, height: height))
    }
}
#endif