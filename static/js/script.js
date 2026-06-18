const config = {
    // Keep browser-side validation aligned with Flask's upload limits.
    maxFileSize: 500 * 1024 * 1024,
    allowedExtensions: ["wav", "mp3", "m4a", "flac", "ogg"],
    endpoints: {
        upload: "/api/upload",
        process: "/api/process",
        downloadTxt: "/api/download/txt",
        downloadPdf: "/api/download/pdf",
        bulkWerStart: "/evaluate-wer-folder/start",
        bulkWerStatus: "/evaluate-wer-folder/status"
    }
};

const initialSession = () => ({
    // One browser session tracks the uploaded file plus the generated transcript.
    sessionId: null,
    filename: null,
    transcript: null,
    uploadedAt: null,
    model: "base",
    device: "cpu"
});

let currentSession = initialSession();
let bulkWerPollTimer = null;

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
    // Bind only the controls present on the current page.
    if ($("uploadForm")) {
        initializeUploadZone();
        $("uploadForm").addEventListener("submit", handleFormSubmit);
    }

    if ($("downloadTxt")) {
        $("downloadTxt").addEventListener("click", () => downloadAs("txt"));
    }

    if ($("downloadPdf")) {
        $("downloadPdf").addEventListener("click", () => downloadAs("pdf"));
    }

    if ($("copyBtn")) {
        $("copyBtn").addEventListener("click", copyTranscript);
    }

    if ($("clearBtn")) {
        $("clearBtn").addEventListener("click", clearTranscript);
    }

    bindEvaluationForms();
    bindBulkWerEvaluation();
});

function bindEvaluationForms() {
    // Evaluation forms post normally, so show a simple loading state while Flask works.
    const forms = document.querySelectorAll("[data-evaluation-form]");
    forms.forEach((form) => {
        form.addEventListener("submit", () => {
            if (form.hasAttribute("data-bulk-wer-form")) {
                return;
            }

            const button = form.querySelector("[data-loading-button]");
            const message = form.querySelector(".loading-message");

            if (button) {
                button.disabled = true;
                button.dataset.originalText = button.textContent;
                button.textContent = "Running Evaluation...";
            }

            if (message) {
                message.hidden = false;
            }
        });
    });
}

function bindBulkWerEvaluation() {
    // Bulk WER can take a long time, so this form starts a background task instead.
    const form = document.querySelector("[data-bulk-wer-form]");
    if (!form) {
        return;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const button = form.querySelector("[data-loading-button]");
        const message = form.querySelector(".loading-message");
        const formData = new FormData(form);

        resetBulkWerUi();

        if (button) {
            button.disabled = true;
            button.dataset.originalText = button.textContent;
            button.textContent = "Starting Evaluation...";
        }

        if (message) {
            message.hidden = false;
        }

        try {
            const response = await fetch(config.endpoints.bulkWerStart, {
                method: "POST",
                body: formData
            });
            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorMessage = data.error || (Array.isArray(data.errors) ? data.errors.join(" ") : "Failed to start bulk WER evaluation.");
                throw new Error(errorMessage);
            }

            showBulkWerProgress({
                percent: data.percent || 0,
                processed_files: data.processed_files || 0,
                total_files: data.total_files || 0,
                current_file: null,
                message: data.message || "Bulk WER evaluation started."
            });

            startBulkWerPolling(data.task_id, button, message);
        } catch (error) {
            if (button) {
                button.disabled = false;
                button.textContent = button.dataset.originalText || "Calculate Bulk WER Accuracy";
            }

            if (message) {
                message.hidden = true;
            }

            showBulkWerErrors([error.message]);
        }
    });
}

function startBulkWerPolling(taskId, button, message) {
    // Poll the Flask task registry until the worker reports completion or failure.
    stopBulkWerPolling();

    const poll = async () => {
        try {
            const response = await fetch(`${config.endpoints.bulkWerStatus}/${encodeURIComponent(taskId)}`);
            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Failed to fetch bulk WER progress.");
            }

            const task = data.task;
            showBulkWerProgress(task);

            if (task.status === "completed") {
                stopBulkWerPolling();
                if (button) {
                    button.disabled = false;
                    button.textContent = button.dataset.originalText || "Calculate Bulk WER Accuracy";
                }
                if (message) {
                    message.hidden = true;
                }

                const taskErrors = Array.isArray(task.errors) ? task.errors.map((item) => item.error || String(item)) : [];
                if (taskErrors.length > 0) {
                    showBulkWerErrors(taskErrors);
                }

                if (task.result) {
                    renderBulkWerResults(task.result);
                }
                return;
            }

            if (task.status === "failed") {
                stopBulkWerPolling();
                if (button) {
                    button.disabled = false;
                    button.textContent = button.dataset.originalText || "Calculate Bulk WER Accuracy";
                }
                if (message) {
                    message.hidden = true;
                }

                const taskErrors = Array.isArray(task.errors) && task.errors.length > 0
                    ? task.errors.map((item) => item.error || String(item))
                    : [task.message || "Bulk WER evaluation failed."];
                showBulkWerErrors(taskErrors);
            }
        } catch (error) {
            stopBulkWerPolling();
            if (button) {
                button.disabled = false;
                button.textContent = button.dataset.originalText || "Calculate Bulk WER Accuracy";
            }
            if (message) {
                message.hidden = true;
            }
            showBulkWerErrors([error.message]);
        }
    };

    poll();
    bulkWerPollTimer = window.setInterval(poll, 1500);
}

function stopBulkWerPolling() {
    // Clear any previous interval before starting a new bulk WER run.
    if (bulkWerPollTimer) {
        window.clearInterval(bulkWerPollTimer);
        bulkWerPollTimer = null;
    }
}

function resetBulkWerUi() {
    // Remove stale dynamic results before a new asynchronous evaluation starts.
    const resultsBox = $("bulkWerDynamicResults");
    const errorsBox = $("bulkWerDynamicErrors");
    const errorsBody = $("bulkWerDynamicErrorsBody");

    if (resultsBox) {
        resultsBox.hidden = true;
        resultsBox.innerHTML = "";
    }

    if (errorsBox) {
        errorsBox.hidden = true;
    }

    if (errorsBody) {
        errorsBody.innerHTML = "";
    }
}

function showBulkWerProgress(task) {
    // Mirror backend progress fields into the inline progress panel.
    const box = $("bulkWerInlineProgress");
    const fill = $("bulkWerProgressFill");
    const progressText = $("bulkWerProgressText");
    const progressCount = $("bulkWerProgressCount");
    const currentFile = $("bulkWerCurrentFile");

    if (!box || !fill || !progressText || !progressCount || !currentFile) {
        return;
    }

    box.hidden = false;
    fill.style.width = `${Math.max(0, Math.min(100, Number(task.percent || 0)))}%`;
    progressText.textContent = `${Math.round(Number(task.percent || 0))}% - ${task.message || "Running bulk WER evaluation"}`;
    progressCount.textContent = `${task.processed_files || 0} / ${task.total_files || 0} files processed`;
    currentFile.textContent = `Current file: ${task.current_file || "none"}`;
}

function showBulkWerErrors(errors) {
    // Render worker errors as plain text to avoid injecting uploaded filenames as HTML.
    const errorsBox = $("bulkWerDynamicErrors");
    const errorsBody = $("bulkWerDynamicErrorsBody");

    if (!errorsBox || !errorsBody) {
        return;
    }

    errorsBody.innerHTML = "";
    (errors || []).forEach((error) => {
        const item = document.createElement("p");
        item.textContent = error;
        errorsBody.appendChild(item);
    });
    errorsBox.hidden = false;
}

function renderBulkWerResults(result) {
    // Build the same result summary the server template renders for synchronous runs.
    const container = $("bulkWerDynamicResults");
    if (!container) {
        return;
    }

    const rows = Array.isArray(result.file_results) ? result.file_results.map((item) => `
        <tr>
            <td>${escapeHtml(item.file_name || "")}</td>
            <td>${escapeHtml(item.wer ?? "")}</td>
            <td>${escapeHtml(item.cer ?? "")}</td>
            <td>${escapeHtml(item.transcription_accuracy ?? "")}</td>
            <td>${escapeHtml(item.status || "")}</td>
        </tr>
    `).join("") : "";

    const reportLink = result.report_csv_path
        ? `<a class="accuracy-btn" href="/evaluation-download?path=${encodeURIComponent(result.report_csv_path)}">Download WER CSV Report</a>`
        : "";

    container.innerHTML = `
        <div class="result-box">
            <div class="success-box">Bulk WER Evaluation Result</div>
            <div class="result-grid">
                <div class="metric-card">
                    <span class="metric-value">${escapeHtml(result.total_files ?? 0)}</span>
                    <span class="metric-label">Total Files</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value success-rate">${escapeHtml(result.successful_files ?? 0)}</span>
                    <span class="metric-label">Successful Files</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value error-rate">${escapeHtml(result.failed_files ?? 0)}</span>
                    <span class="metric-label">Failed Files</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value error-rate">${escapeHtml(result.average_wer ?? 0)}%</span>
                    <span class="metric-label">Average WER</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value error-rate">${escapeHtml(result.average_cer ?? 0)}%</span>
                    <span class="metric-label">Average CER</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value success-rate">${escapeHtml(result.average_transcription_accuracy ?? 0)}%</span>
                    <span class="metric-label">Average Transcription Accuracy</span>
                </div>
            </div>
            ${reportLink}
            <div class="table-wrap">
                <table class="result-table">
                    <thead>
                        <tr>
                            <th>File Name</th>
                            <th>WER %</th>
                            <th>CER %</th>
                            <th>Transcription Accuracy %</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
    container.hidden = false;
}

function initializeUploadZone() {
    // Wire click, keyboard, and drag/drop interactions to the hidden file input.
    const dropZone = $("dropZone");
    const fileInput = $("fileInput");

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        const [file] = event.dataTransfer.files;
        if (file) {
            fileInput.files = event.dataTransfer.files;
            handleFileSelection(file);
        }
    });
}

function handleFileSelection(file) {
    // Validate size and extension before uploading to save a round trip.
    const extension = file.name.split(".").pop().toLowerCase();
    const fileSizeMb = (file.size / (1024 * 1024)).toFixed(2);

    hideError();

    if (!config.allowedExtensions.includes(extension)) {
        showError(`Unsupported file format: ${extension}. Use ${config.allowedExtensions.join(", ")}.`);
        resetFileSelection();
        return;
    }

    if (file.size > config.maxFileSize) {
        showError(`File is too large: ${fileSizeMb} MB. Maximum allowed size is 500 MB.`);
        resetFileSelection();
        return;
    }

    $("selectedFileName").textContent = file.name;
    $("selectedFileSize").textContent = `${fileSizeMb} MB`;
    $("fileInfo").hidden = false;
    $("uploadBtn").disabled = false;
}

async function handleFormSubmit(event) {
    // Upload first, then ask the backend to run transcription and diarization.
    event.preventDefault();

    const file = $("fileInput").files[0];
    const model = $("modelSelect").value;
    const device = $("deviceSelect").value;

    if (!file) {
        showError("Select an audio file before starting.");
        return;
    }

    showProcessingSection(model, device);
    setStepState("upload", "active");
    updateProgress(8, "Uploading audio");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const uploadResponse = await fetch(config.endpoints.upload, {
            method: "POST",
            body: formData
        });
        const uploadData = await uploadResponse.json();

        if (!uploadResponse.ok || !uploadData.success) {
            throw new Error(uploadData.error || "Upload failed.");
        }

        currentSession.sessionId = uploadData.session_id;
        currentSession.filename = uploadData.filepath;
        currentSession.uploadedAt = new Date();
        currentSession.model = model;
        currentSession.device = device;

        setStepState("upload", "completed");
        setStepState("whisper", "active");
        updateProgress(26, "Upload complete");

        await startProcessing(uploadData.session_id, uploadData.filepath, model, device);
    } catch (error) {
        showError(error.message);
    }
}

async function startProcessing(sessionId, filename, model, device) {
    // The backend performs Whisper, Pyannote, and merging in one request.
    updateProgress(42, `Running Whisper (${model})`);

    try {
        const response = await fetch(config.endpoints.process, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                filename,
                model,
                device
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Processing failed.");
        }

        currentSession.transcript = Array.isArray(data.transcript) ? data.transcript : [];
        currentSession.model = data.model || model;
        currentSession.device = data.device || device;

        setStepState("whisper", "completed");
        setStepState("diarization", "completed");
        setStepState("merge", "completed");
        updateProgress(100, "Transcript ready");

        renderTranscript(currentSession.transcript);
        updateRunBadges(currentSession.model, currentSession.device);
        showTranscriptSection();
    } catch (error) {
        showError(error.message);
    }
}

function updateProgress(percent, message) {
    // Keep the progress bar and status copy in sync.
    $("progressFill").style.width = `${percent}%`;
    $("progressText").textContent = `${Math.round(percent)}% - ${message}`;
    $("statusMessage").textContent = message;
}

function setStepState(stepId, state) {
    // Steps use CSS classes for active/completed visual state.
    const step = $(`step-${stepId}`);
    if (!step) {
        return;
    }

    step.classList.remove("active", "completed");

    if (state) {
        step.classList.add(state);
    }
}

function renderTranscript(transcriptData) {
    // Render transcript rows with escaped text so audio content cannot inject markup.
    const container = $("transcriptContent");
    container.innerHTML = "";

    transcriptData.forEach((entry) => {
        const row = document.createElement("article");
        row.className = "transcript-entry";
        row.innerHTML = `
            <span class="timestamp">${escapeHtml(entry.timestamp || "00:00:00")}</span>
            <span class="speaker">${escapeHtml(entry.speaker || "Speaker")}</span>
            <span class="transcript-text">${escapeHtml(entry.text || "")}</span>
        `;
        container.appendChild(row);
    });

    $("summaryEntries").textContent = String(transcriptData.length);
    $("summaryModel").textContent = currentSession.model;
    $("summaryDevice").textContent = formatDeviceLabel(currentSession.device);
}

async function downloadAs(format) {
    // TXT is generated in the browser; PDF is generated by Flask/ReportLab.
    if (!currentSession.transcript || currentSession.transcript.length === 0) {
        showError("There is no transcript available to download.");
        return;
    }

    if (format === "txt") {
        downloadTxt(currentSession.transcript);
        return;
    }

    if (format === "pdf") {
        await downloadPdf(currentSession.transcript);
    }
}

function downloadTxt(transcriptData) {
    // Create a lightweight text export without another server request.
    let content = "Meeting Transcript\n";
    content += `${"=".repeat(32)}\n`;
    content += `Generated: ${new Date().toLocaleString()}\n`;
    content += `Session ID: ${currentSession.sessionId}\n`;
    content += `Model: ${currentSession.model}\n`;
    content += `Device: ${formatDeviceLabel(currentSession.device)}\n`;
    content += `${"=".repeat(32)}\n\n`;

    transcriptData.forEach((entry) => {
        content += `[${entry.timestamp}]\n`;
        content += `${entry.speaker}\n`;
        content += `${entry.text}\n\n`;
    });

    downloadFile(content, "meeting_transcript.txt", "text/plain");
}

function downloadFile(content, filename, mimeType) {
    // Use a temporary object URL to trigger browser downloads.
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

async function downloadPdf(transcriptData) {
    // Send transcript rows to Flask so ReportLab can produce a PDF.
    try {
        const response = await fetch(config.endpoints.downloadPdf, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: currentSession.sessionId,
                model: currentSession.model,
                device: formatDeviceLabel(currentSession.device),
                transcript: transcriptData
            })
        });

        if (!response.ok) {
            let message = "PDF download failed.";
            const contentType = response.headers.get("content-type") || "";
            if (contentType.includes("application/json")) {
                const data = await response.json();
                message = data.error || message;
            }
            throw new Error(message);
        }

        const blob = await response.blob();
        downloadBlob(blob, "meeting_transcript.pdf");
    } catch (error) {
        showError(error.message || "PDF download failed.");
    }
}

function downloadBlob(blob, filename) {
    // Download binary responses such as generated PDFs.
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function copyTranscript() {
    // Copy the rendered transcript text, preserving the visible speaker order.
    if (!currentSession.transcript || currentSession.transcript.length === 0) {
        showError("There is no transcript to copy.");
        return;
    }

    const content = $("transcriptContent").innerText;
    navigator.clipboard.writeText(content).then(() => {
        const button = $("copyBtn");
        const originalText = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => {
            button.textContent = originalText;
        }, 1600);
    }).catch(() => {
        showError("Clipboard access failed.");
    });
}

function clearTranscript() {
    // Reset all UI state so the next run starts from a clean upload form.
    if (!window.confirm("Reset the current transcript and start over?")) {
        return;
    }

    currentSession = initialSession();

    $("uploadForm").reset();
    $("uploadBtn").disabled = true;
    $("fileInfo").hidden = true;
    $("statusSection").hidden = true;
    $("transcriptSection").hidden = true;
    $("uploadSection").hidden = false;
    $("transcriptContent").innerHTML = "";
    $("progressFill").style.width = "0%";
    $("progressText").textContent = "0% - Waiting to start";
    $("statusMessage").textContent = "Upload a file to begin.";

    ["upload", "whisper", "diarization", "merge"].forEach((stepId) => setStepState(stepId, ""));

    updateRunBadges("base", "cpu");
    $("summaryEntries").textContent = "0";
    $("summaryModel").textContent = "base";
    $("summaryDevice").textContent = "CPU";
    hideError();
}

function showProcessingSection(model, device) {
    // Switch from the upload form to progress tracking.
    $("uploadSection").hidden = true;
    $("statusSection").hidden = false;
    $("transcriptSection").hidden = true;
    updateRunBadges(model, device);
    hideError();
}

function showTranscriptSection() {
    // Reveal the final transcript after rendering has completed.
    $("transcriptSection").hidden = false;
}

function updateRunBadges(model, device) {
    // Show the effective model and device returned by the backend.
    $("activeModelBadge").textContent = `Model: ${model}`;
    $("activeDeviceBadge").textContent = `Device: ${formatDeviceLabel(device)}`;
}

function formatDeviceLabel(device) {
    // CUDA is available only on compatible GPU systems; CPU is the Mac-safe default.
    return device && device.toLowerCase() === "cuda" ? "GPU (CUDA)" : "CPU";
}

function showError(message) {
    // Surface backend and browser validation failures in the status panel.
    $("errorText").textContent = message;
    $("errorMessage").hidden = false;
    $("statusMessage").textContent = message;
}

function hideError() {
    // Clear stale errors before the next user action.
    $("errorMessage").hidden = true;
    $("errorText").textContent = "";
}

function resetFileSelection() {
    // Remove an invalid file from the input and disable processing.
    $("fileInput").value = "";
    $("fileInfo").hidden = true;
    $("uploadBtn").disabled = true;
}

function escapeHtml(text) {
    // Escape dynamic values before inserting them into HTML strings.
    const value = String(text);
    const map = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#039;"
    };

    return value.replace(/[&<>"']/g, (character) => map[character]);
}
