import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var latestScan: LatestScan?
    @Published var modelMetadata: ModelMetadata?
    @Published var trainingEvents: [TrainingEvent] = []
    @Published var scanStages: [ScanStage] = ScanStage.defaults
    @Published var currentJob: String = "Idle"
    @Published var currentDetail: String = "Ready"
    @Published var progress: Double = 0
    @Published var isRunning = false
    @Published var elapsedSeconds: Int = 0
    @Published var autoTrainerEnabled = true
    @Published var selectedUniverse = "full"
    @Published var useFreshData = false
    @Published var lastError: String?

    private var jobStartDate: Date = .distantPast
    private var elapsedTimer: Timer?

    let projectRoot: URL

    init() {
        let envRoot = ProcessInfo.processInfo.environment["STOCK_PREDICTOR_ROOT"]
        if let envRoot, !envRoot.isEmpty {
            projectRoot = URL(fileURLWithPath: envRoot)
        } else {
            // When run as a .app bundle, walk up from the bundle to find the project root.
            // Fall back to the known absolute path on disk if the bundle-relative path has no venv.
            let bundleDerived = Bundle.main.bundleURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            let knownPath = "/Users/crinklyink/Desktop/Stock scan"
            let knownURL = URL(fileURLWithPath: knownPath)
            let venvCheck = knownURL.appendingPathComponent(".venv/bin/python3").path
            if FileManager.default.isExecutableFile(atPath: venvCheck) {
                projectRoot = knownURL
            } else {
                projectRoot = bundleDerived
            }
        }
    }

    var candidates: [Candidate] {
        latestScan?.candidates ?? []
    }

    var summary: ScanSummary? {
        latestScan?.scanSummary
    }

    var scanUsesOlderModel: Bool {
        guard let trainedFamily = modelMetadata?.modelFamily else { return false }
        guard let scanFamily = latestScan?.trainingReport?.modelFamily else {
            return latestScan != nil
        }
        return scanFamily != trainedFamily
    }

    var activeModelSummary: String {
        modelMetadata?.modeSummary ?? "No trained model"
    }

    var scanGeneratedText: String {
        guard let generated = latestScan?.generatedAt else { return "No scan artifact" }
        return String(generated.prefix(16)).replacingOccurrences(of: "T", with: " ")
    }

    var syncStatusText: String {
        scanUsesOlderModel ? "Scan artifact needs refresh" : "Model and scan aligned"
    }

    func reload() async {
        // Clear any prior decode errors before reloading
        lastError = nil
        latestScan = loadJSON("stock_predictor/artifacts/latest_scan.json", as: LatestScan.self)
        let adaptive = loadJSON("stock_predictor/models/adaptive_model_metadata.json", as: ModelMetadata.self)
        let xgb = loadJSON("stock_predictor/models/xgboost_metadata.json", as: ModelMetadata.self)
        let reportMetadata = latestScan?.trainingReport.map(ModelMetadata.init(report:))
        modelMetadata = Self.preferredModelMetadata(xgb: xgb, adaptive: adaptive, report: reportMetadata)
        // If the scan loaded fine, clear any decode error that might have remained
        if latestScan != nil { lastError = nil }
        loadEnvSettings()
    }

    private nonisolated static func preferredModelMetadata(
        xgb: ModelMetadata?,
        adaptive: ModelMetadata?,
        report: ModelMetadata?
    ) -> ModelMetadata? {
        if xgb?.modelFamily == "RandomForestFallback" || xgb?.labelDefinition?.contains("+6%") == true {
            return xgb
        }
        return adaptive ?? xgb ?? report
    }

    func runTraining() {
        runBackendJob(flag: "--train", title: "Training model")
    }

    func runBacktest() {
        runBackendJob(flag: "--backtest-adaptive", title: "Rolling backtest")
    }

    func runScan() {
        // --top-n 10 runs a full scan and returns top 10 candidates; --paper-trade logs them
        runBackendJob(flag: "--top-n", title: "Weekly scan", extra: ["10", "--paper-trade"])
    }

    func setupPythonEnvironment() {
        guard !isRunning else { return }
        isRunning = true
        progress = 0.02
        elapsedSeconds = 0
        jobStartDate = Date()
        currentJob = "Setting up Python"
        currentDetail = "Creating local .venv"
        trainingEvents.removeAll()
        scanStages = ScanStage.defaults
        let root = projectRoot
        Task.detached(priority: .userInitiated) {
            let venvPython = root.appendingPathComponent(".venv/bin/python3").path
            let steps: [(String, [String], String, Double)] = [
                ("/usr/bin/env", ["python3", "-m", "venv", ".venv"], "Creating .venv", 0.18),
                (venvPython, ["-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip", 0.42),
                (venvPython, ["-m", "pip", "install", "-r", "requirements.txt"], "Installing requirements", 0.72),
            ]
            var allOutput: [String] = []
            for (executable, arguments, detail, progress) in steps {
                await MainActor.run {
                    self.currentDetail = detail
                    self.progress = progress
                }
                let result = await self.runProcess(executable: executable, arguments: arguments, cwd: root)
                allOutput.append(contentsOf: result.output)
                if result.exitCode != 0 {
                    await MainActor.run {
                        let tail = allOutput.suffix(8).joined(separator: " ")
                        self.lastError = "Environment setup failed: \(tail)"
                        self.finishJob(status: "Failed", detail: "Environment setup failed")
                    }
                    return
                }
            }
            await MainActor.run {
                self.lastError = nil
                self.progress = 1.0
                self.finishJob(status: "Complete", detail: "Local .venv is ready")
            }
        }
    }

    func toggleAutoTrainer(_ enabled: Bool) {
        autoTrainerEnabled = enabled
        writeEnvValue(key: "RETRAIN_MODEL_WEEKLY", value: enabled ? "1" : "0")
    }

    func setDefaultUniverse(_ universe: String) {
        selectedUniverse = universe
        writeEnvValue(key: "DEFAULT_UNIVERSE", value: universe)
    }

    private func runBackendJob(flag: String, title: String, extra: [String] = []) {
        guard !isRunning else { return }
        isRunning = true
        progress = 0.04
        elapsedSeconds = 0
        jobStartDate = Date()
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRunning else { return }
                self.elapsedSeconds = Int(Date().timeIntervalSince(self.jobStartDate))
            }
        }
        currentJob = title
        currentDetail = "Starting"
        trainingEvents.removeAll()
        scanStages = ScanStage.defaults
        markScanStage("fetching_data", status: .active)

        let root = projectRoot
        let universe = selectedUniverse
        let fresh = useFreshData
        let pythonPath = backendPythonExecutable(projectRoot: root)

        Task.detached(priority: .userInitiated) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: pythonPath)
            var arguments = ["main.py", flag] + extra + ["--universe", universe]
            if fresh { arguments.append("--fresh") }
            process.arguments = arguments
            process.currentDirectoryURL = root
            process.environment = ProcessInfo.processInfo.environment.merging(
                [
                    "PYTHONUNBUFFERED": "1",
                    "STOCK_PREDICTOR_ROOT": root.path,
                    "STOCK_PREDICTOR_PYTHON": pythonPath,
                    "STOCK_PREDICTOR_QUIET_RUNTIME": "1",
                    "PROGRESS_BAR": "0",
                    "ENABLE_FINBERT_SENTIMENT": "0",
                    "ADAPTIVE_MODEL": "0",
                    "NATIVE_BOOSTERS": "0",
                    "LIGHTGBM_ENSEMBLE": "0",
                    "HYPERPARAMETER_SEARCH": "0",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                ],
                uniquingKeysWith: { _, new in new }
            )

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                await self.consume(text: "Using Python: \(pythonPath)")
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.finishJob(status: "Failed", detail: "Could not start: \(error.localizedDescription)")
                }
                return
            }

            let handle = pipe.fileHandleForReading
            var outputLines: [String] = []
            while process.isRunning {
                let data = handle.availableData
                if data.isEmpty { break }
                outputLines.append(contentsOf: Self.lines(from: data))
                await self.consume(data: data)
            }
            let remaining = handle.readDataToEndOfFile()
            if !remaining.isEmpty {
                await self.consume(data: remaining)
                outputLines.append(contentsOf: Self.lines(from: remaining))
            }
            process.waitUntilExit()

            await MainActor.run {
                if process.terminationStatus == 0 {
                    self.lastError = nil
                    self.completeAllScanStages()
                    self.finishJob(status: "Complete", detail: "Artifacts refreshed")
                    Task { await self.reload() }
                } else {
                    let tail = outputLines.suffix(5).joined(separator: " ")
                    let detail = self.backendFailureMessage(exitCode: process.terminationStatus, tail: tail)
                    self.lastError = detail
                    self.finishJob(status: "Failed", detail: detail)
                }
            }
        }
    }

    private func backendPythonExecutable(projectRoot: URL) -> String {
        let envPython = ProcessInfo.processInfo.environment["STOCK_PREDICTOR_PYTHON"]
        // Always try the hardcoded project venv first; this is the one with all packages installed.
        let knownVenv = "/Users/crinklyink/Desktop/Stock scan/.venv/bin/python3"
        let candidates = [
            envPython,
            knownVenv,
            projectRoot.appendingPathComponent(".venv/bin/python3").path,
            projectRoot.appendingPathComponent(".venv/bin/python").path,
            "/opt/homebrew/bin/python3.13",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3"
        ].compactMap { $0 }.filter { !$0.isEmpty }
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) } ?? "/usr/bin/python3"
    }

    private nonisolated func runProcess(executable: String, arguments: [String], cwd: URL) async -> (exitCode: Int32, output: [String]) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = cwd
        process.environment = ProcessInfo.processInfo.environment.merging(
            [
                "PYTHONUNBUFFERED": "1",
                "STOCK_PREDICTOR_ROOT": cwd.path,
                "STOCK_PREDICTOR_QUIET_RUNTIME": "1",
                "PROGRESS_BAR": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            ],
            uniquingKeysWith: { _, new in new }
        )
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
        } catch {
            await consume(text: "Could not start \(executable): \(error.localizedDescription)")
            return (127, [])
        }
        var output: [String] = []
        let handle = pipe.fileHandleForReading
        while process.isRunning {
            let data = handle.availableData
            if data.isEmpty { break }
            let lines = Self.lines(from: data)
            output.append(contentsOf: lines)
            await consume(lines: lines)
        }
        let remaining = handle.readDataToEndOfFile()
        if !remaining.isEmpty {
            let lines = Self.lines(from: remaining)
            output.append(contentsOf: lines)
            await consume(lines: lines)
        }
        process.waitUntilExit()
        return (process.terminationStatus, output)
    }

    private nonisolated static func lines(from data: Data) -> [String] {
        guard let chunk = String(data: data, encoding: .utf8) else { return [] }
        return chunk
            .split { $0.isNewline || $0 == "\r" }
            .map(String.init)
            .filter {
                let trimmed = $0.trimmingCharacters(in: .whitespaces)
                return !trimmed.isEmpty
                    && !trimmed.hasPrefix("Scoring tickers:")
                    && !trimmed.hasPrefix("Final scoring:")
            }
    }

    private func backendFailureMessage(exitCode: Int32, tail: String) -> String {
        if tail.contains("ModuleNotFoundError") || tail.contains("No module named") {
            return "Python dependencies are missing. Create .venv and install requirements.txt, or set STOCK_PREDICTOR_PYTHON."
        }
        if exitCode == 11 && (tail.localizedCaseInsensitiveContains("huggingface") || tail.localizedCaseInsensitiveContains("finbert")) {
            return "The optional FinBERT sentiment model crashed while loading. FinBERT is disabled for app runs now; run the scan again."
        }
        if tail.isEmpty {
            return "Backend exited with code \(exitCode). No output was returned."
        }
        return "Backend exited with code \(exitCode): \(tail)"
    }

    private func consume(data: Data) async {
        let lines = Self.lines(from: data)
        await MainActor.run {
            for line in lines {
                trainingEvents.append(TrainingEvent(text: line))
                updateProgress(from: line)
            }
            trimTrainingEvents()
        }
    }

    private func consume(text: String) async {
        await MainActor.run {
            trainingEvents.append(TrainingEvent(text: text))
            trimTrainingEvents()
        }
    }

    private func consume(lines: [String]) async {
        await MainActor.run {
            for line in lines {
                trainingEvents.append(TrainingEvent(text: line))
                updateProgress(from: line)
            }
            trimTrainingEvents()
        }
    }

    private func trimTrainingEvents() {
        let limit = 260
        if trainingEvents.count > limit {
            trainingEvents.removeFirst(trainingEvents.count - limit)
        }
    }

    private func updateProgress(from line: String) {
        let lower = line.lowercased()
        if lower.contains("resolving") || lower.contains("fetching historical") {
            progress = max(progress, 0.26)
            markScanStage("fetching_data", status: .complete)
            markScanStage("scoring", status: .active)
        }
        if lower.contains("sector") || lower.contains("macro") || lower.contains("benchmark") {
            progress = max(progress, 0.44)
            markScanStage("scoring", status: .complete)
            markScanStage("filtering", status: .active)
        }
        if lower.contains("calculating") || lower.contains("threshold") {
            progress = max(progress, 0.54)
            markScanStage("filtering", status: .complete)
            markScanStage("ranking", status: .active)
        }
        if lower.contains("training adaptive") {
            progress = max(progress, 0.70)
            markScanStage("ranking", status: .active)
        }
        if lower.contains("walk-forward") || lower.contains("rolling backtest") {
            progress = max(progress, 0.82)
            markScanStage("ranking", status: .complete)
            markScanStage("saving", status: .active)
        }
        if lower.contains("saving") || lower.contains("writing") {
            progress = max(progress, 0.93)
            markScanStage("saving", status: .active)
        }
        if lower.contains("complete") {
            progress = 1.0
            completeAllScanStages()
        }
        if line.hasPrefix("[training]") || line.hasPrefix("[backtest]") {
            currentDetail = line
                .replacingOccurrences(of: "[training] ", with: "")
                .replacingOccurrences(of: "[backtest] ", with: "")
        }
    }

    private func markScanStage(_ id: String, status: ScanStageStatus) {
        guard let index = scanStages.firstIndex(where: { $0.id == id }) else { return }
        scanStages[index].status = status
        scanStages[index].detail = currentDetail
        if status == .active {
            for prior in scanStages.indices where scanStages[prior].order < scanStages[index].order && scanStages[prior].status == .pending {
                scanStages[prior].status = .complete
            }
        }
    }

    private func completeAllScanStages() {
        for index in scanStages.indices {
            scanStages[index].status = .complete
        }
    }

    private func finishJob(status: String, detail: String) {
        currentJob = status
        currentDetail = detail
        progress = status == "Complete" ? 1.0 : progress
        isRunning = false
        elapsedTimer?.invalidate()
        elapsedTimer = nil
        // Reset status label after 4 seconds so sidebar shows "Idle" again
        DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in
            guard let self, !self.isRunning else { return }
            self.currentJob = "Idle"
            self.currentDetail = "Ready"
            self.elapsedSeconds = 0
        }
    }

    private func loadJSON<T: Decodable>(_ relativePath: String, as type: T.Type) -> T? {
        let url = projectRoot.appendingPathComponent(relativePath)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        do {
            let result = try decoder.decode(T.self, from: data)
            // Clear any prior error for this file now that it decoded successfully
            if lastError?.contains(relativePath) == true { lastError = nil }
            return result
        } catch {
            // Only surface errors for the primary scan file; ignore metadata decode noise
            if relativePath.contains("latest_scan") {
                lastError = "Could not read \(relativePath): \(error.localizedDescription)"
            }
            return nil
        }
    }

    private func loadEnvSettings() {
        let url = projectRoot.appendingPathComponent(".env")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            if parts[0] == "RETRAIN_MODEL_WEEKLY" {
                autoTrainerEnabled = parts[1] != "0"
            }
            if parts[0] == "DEFAULT_UNIVERSE" {
                selectedUniverse = parts[1]
            }
        }
    }

    private func writeEnvValue(key: String, value: String) {
        let url = projectRoot.appendingPathComponent(".env")
        var lines = (try? String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)) ?? []
        var found = false
        for index in lines.indices {
            if lines[index].hasPrefix("\(key)=") {
                lines[index] = "\(key)=\(value)"
                found = true
            }
        }
        if !found { lines.append("\(key)=\(value)") }
        try? (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

extension ModelMetadata {
    init(report: TrainingReport) {
        trained = report.trained
        trainedAt = report.trainedAt
        trainingSamples = report.trainingSamples
        validationSamples = report.validationSamples
        modelFamily = report.modelFamily
        selectedProfile = report.selectedProfile
        auc = report.auc
        ensembleAuc = report.ensembleAuc
        xgbAuc = report.xgbAuc
        lightgbmAuc = report.lightgbmAuc
        shortHorizonAuc = report.shortHorizonAuc
        labelDefinition = report.labelDefinition
        ensembleWeights = nil
        featureImportance = report.featureImportance
    }
}
