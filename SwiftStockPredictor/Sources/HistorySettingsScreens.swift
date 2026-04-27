import SwiftUI

struct HistoryScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        PageFrame {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("History", subtitle: "Rolling paper-trade and backtest reporting from the existing backend artifacts.")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                MetricTile(title: "Weeks", value: "\(appState.latestScan?.paperTradeSummary?.weeks ?? 0)", footnote: "tracked")
                MetricTile(title: "Hit Rate", value: String(format: "%.0f%%", appState.latestScan?.paperTradeSummary?.targetHitRate ?? 0), footnote: "target hits", tone: Design.green)
                MetricTile(title: "Positive", value: String(format: "%.0f%%", appState.latestScan?.paperTradeSummary?.positiveReturnRate ?? 0), footnote: "green closes")
                MetricTile(title: "Avg Return", value: (appState.latestScan?.paperTradeSummary?.averageReturn ?? 0).percentText, footnote: "per pick")
            }
            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Scan History Timeline")
                        .font(.headline)
                    ForEach(historyRows, id: \.date) { row in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(spacing: 4) {
                                Circle()
                                    .fill(row.tone)
                                    .frame(width: 12, height: 12)
                                Rectangle()
                                    .fill(.secondary.opacity(0.18))
                                    .frame(width: 2, height: 34)
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(row.date)
                                        .font(.callout.monospacedDigit().weight(.semibold))
                                    Pill(text: row.regime, color: row.tone)
                                    Spacer()
                                    Text(row.returnText)
                                        .font(.callout.monospacedDigit().weight(.semibold))
                                        .foregroundStyle(row.tone)
                                }
                                Text(row.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    HStack {
                        Pill(text: "Best \(appState.latestScan?.paperTradeSummary?.bestPick ?? "--") \(String(format: "%+.1f%%", appState.latestScan?.paperTradeSummary?.bestReturn ?? 0))", color: Design.green)
                        Pill(text: "Worst \(appState.latestScan?.paperTradeSummary?.worstPick ?? "--") \(String(format: "%+.1f%%", appState.latestScan?.paperTradeSummary?.worstReturn ?? 0))", color: Design.red)
                        Spacer()
                    }
                    Divider().opacity(0.35)
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

    private var historyRows: [(date: String, regime: String, detail: String, returnText: String, tone: Color)] {
        let rows = appState.latestScan?.backtestRows ?? []
        let grouped = Dictionary(grouping: rows) { row in
            String((row.createdAt ?? row.runID).prefix(10))
        }
        let artifactRows = grouped.map { date, rows in
            let sorted = rows.sorted { ($0.finalScore ?? 0) > ($1.finalScore ?? 0) }
            let top = sorted.prefix(3).map(\.ticker).joined(separator: ", ")
            let avgReturn = rows.compactMap(\.realizedReturn).map { $0 * 100.0 }.average
            let hitRate = rows.compactMap(\.resolvedTargetHit).average * 100.0
            let regime = appState.latestScan?.regimeLabel ?? appState.summary?.regime ?? "unknown"
            let detail = "Top picks: \(top). Hit rate \(String(format: "%.0f%%", hitRate)); \(rows.count) tracked outcomes."
            return (
                date: date,
                regime: regime,
                detail: detail,
                returnText: avgReturn.percentText,
                tone: avgReturn >= 0 ? Design.green : Design.red
            )
        }
        .sorted { $0.date > $1.date }
        .prefix(8)

        if !artifactRows.isEmpty {
            return Array(artifactRows)
        }

        let summary = appState.latestScan?.paperTradeSummary
        let current = appState.scanGeneratedText
        let top = appState.candidates.prefix(3).map(\.ticker).joined(separator: ", ")
        return [(current, appState.latestScan?.regimeLabel ?? "risk_on", "Top picks: \(top). Hit rate \(String(format: "%.0f%%", summary?.targetHitRate ?? 0)).", (summary?.averageReturn ?? 0).percentText, Design.green)]
    }
}

private extension Array where Element == Double {
    var average: Double {
        isEmpty ? 0 : reduce(0, +) / Double(count)
    }
}

struct SettingsScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        PageFrame {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader("Settings", subtitle: "Native app preferences for training, scan universe, and backend execution.")
            GlassPanel {
                VStack(alignment: .leading, spacing: 16) {
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
                    Divider()
                    DataRow(title: "Active model", value: appState.activeModelSummary, tone: Design.blue)
                    DataRow(title: "Scan status", value: appState.syncStatusText, tone: appState.scanUsesOlderModel ? Design.yellow : Design.green)
                    DataRow(title: "Project root", value: appState.projectRoot.path)
                }
            }
            Spacer()
        }
        }
    }
}
