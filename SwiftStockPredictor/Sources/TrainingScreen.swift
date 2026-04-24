import SwiftUI

enum TrainingMode {
    case training
    case backtesting
}

struct TrainingScreen: View {
    @EnvironmentObject private var appState: AppState
    let mode: TrainingMode

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader(
                mode == .training ? "Training" : "Backtesting",
                subtitle: mode == .training
                    ? "Train better models while watching every backend stage."
                    : "Run rolling walk-forward tests and save regime performance reports."
            )
            StatusStrip(
                title: mode == .training ? "Model lab" : "Rolling validation",
                detail: mode == .training
                    ? "Uses the adaptive regime ensemble pipeline, existing market cache, and persisted model artifacts."
                    : "Runs a rolling walk-forward test across regimes and writes the backend report artifact.",
                symbol: mode == .training ? "cpu" : "chart.line.uptrend.xyaxis",
                color: mode == .training ? .blue : .purple
            )

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 4), spacing: 14) {
                MetricTile(title: "Stack", value: appState.modelMetadata?.stackName ?? "--", footnote: appState.modelMetadata?.selectedProfile?.cleanedProfile ?? "profile")
                MetricTile(title: "AUC", value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0), footnote: "walk-forward", tone: Design.green)
                MetricTile(title: "Samples", value: "\(appState.modelMetadata?.trainingSamples ?? 0)", footnote: "training rows")
                MetricTile(title: "Status", value: appState.currentJob, footnote: appState.currentDetail, tone: appState.isRunning ? .blue : .primary)
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(alignment: .center, spacing: 16) {
                        Picker("Universe", selection: Binding(
                            get: { appState.selectedUniverse },
                            set: { appState.setDefaultUniverse($0) }
                        )) {
                            Text("Mini").tag("mini")
                            Text("Full").tag("full")
                            Text("US Market").tag("us_market")
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 360)

                        Toggle("Fresh data", isOn: $appState.useFreshData)
                        Toggle("Auto-train weekly", isOn: Binding(
                            get: { appState.autoTrainerEnabled },
                            set: { appState.toggleAutoTrainer($0) }
                        ))

                        Spacer()

                        Button {
                            mode == .training ? appState.runTraining() : appState.runBacktest()
                        } label: {
                            Label(mode == .training ? "Train Model" : "Run Backtest", systemImage: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(appState.isRunning)
                    }

                    ProgressView(value: appState.progress)
                        .tint(.blue)
                    HStack {
                        Label(appState.currentDetail, systemImage: appState.isRunning ? "arrow.triangle.2.circlepath" : "checkmark.circle")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(appState.progress * 100))%")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        SectionHeader("Live Output")
                        Spacer()
                        Pill(text: appState.isRunning ? "Running" : "Idle", color: appState.isRunning ? .blue : .secondary)
                    }
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(appState.trainingEvents) { event in
                                    Text(event.text)
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundStyle(event.text.hasPrefix("[") ? .primary : .secondary)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .id(event.id)
                                }
                            }
                            .padding(12)
                        }
                        .background(.black.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
                        .onChange(of: appState.trainingEvents.count) {
                            if let last = appState.trainingEvents.last {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }
        }
    }
}
