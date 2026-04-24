// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftStockPredictor",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "StockPredictorSwift", targets: ["StockPredictorSwift"])
    ],
    targets: [
        .executableTarget(
            name: "StockPredictorSwift",
            path: "Sources"
        )
    ]
)
