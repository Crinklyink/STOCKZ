import SwiftUI

@main
struct StockPredictorSwiftApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("Stock Predictor") {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 1180, minHeight: 760)
                .task {
                    await appState.reload()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Stock Predictor") {
                Button("Refresh Data") {
                    Task { await appState.reload() }
                }
                .keyboardShortcut("r", modifiers: [.command])
            }
        }
    }
}
