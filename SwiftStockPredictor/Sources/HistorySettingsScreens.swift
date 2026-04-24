import SwiftUI

struct HistoryScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("History", subtitle: "Rolling paper-trade and backtest reporting from the existing backend artifacts.")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 4), spacing: 14) {
                MetricTile(title: "Weeks", value: "\(appState.latestScan?.paperTradeSummary?.weeks ?? 0)", footnote: "tracked")
                MetricTile(title: "Hit Rate", value: String(format: "%.0f%%", appState.latestScan?.paperTradeSummary?.targetHitRate ?? 0), footnote: "target hits", tone: Design.green)
                MetricTile(title: "Positive", value: String(format: "%.0f%%", appState.latestScan?.paperTradeSummary?.positiveReturnRate ?? 0), footnote: "green closes")
                MetricTile(title: "Avg Return", value: (appState.latestScan?.paperTradeSummary?.averageReturn ?? 0).percentText, footnote: "per pick")
            }
            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Rolling data backtesting")
                        .font(.headline)
                    Text("Use the Backtesting tab to run the adaptive walk-forward test. Reports are saved by the Python backend and reflected here after refresh.")
                        .foregroundStyle(.secondary)
                    HStack {
                        Pill(text: "Best \(appState.latestScan?.paperTradeSummary?.bestPick ?? "--") \(String(format: "%+.1f%%", appState.latestScan?.paperTradeSummary?.bestReturn ?? 0))", color: Design.green)
                        Pill(text: "Worst \(appState.latestScan?.paperTradeSummary?.worstPick ?? "--") \(String(format: "%+.1f%%", appState.latestScan?.paperTradeSummary?.worstReturn ?? 0))", color: Design.red)
                        Spacer()
                    }
                    Button {
                        appState.runBacktest()
                    } label: {
                        Label("Run Rolling Backtest", systemImage: "clock.arrow.circlepath")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(appState.isRunning)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Spacer()
        }
    }
}

struct SettingsScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Settings", subtitle: "Native app preferences for training, scan universe, and backend execution.")
            GlassPanel {
                Form {
                    Picker("Default universe", selection: Binding(
                        get: { appState.selectedUniverse },
                        set: { appState.setDefaultUniverse($0) }
                    )) {
                        Text("Mini").tag("mini")
                        Text("Full").tag("full")
                        Text("US Market").tag("us_market")
                    }
                    Toggle("Use fresh data for jobs", isOn: $appState.useFreshData)
                    Toggle("Auto-train weekly", isOn: Binding(
                        get: { appState.autoTrainerEnabled },
                        set: { appState.toggleAutoTrainer($0) }
                    ))
                    LabeledContent("Project root", value: appState.projectRoot.path)
                }
                .formStyle(.grouped)
            }
            Spacer()
        }
    }
}
