let mode = "";

function chooseMode(selectedMode) {
    mode = selectedMode;

    // Clear previous results
    document.getElementById("result").innerText = "";
    document.getElementById("suggestion").innerText = "";
    
    // Reset button text
    document.getElementById("analyzeButton").innerText = "Analyze Emotion";
    document.getElementById("analyzeButton").disabled = false;

    // Show input area
    document.getElementById("inputArea").style.display = "block";

    // Show/hide text input field based on mode
    if (mode === "text") {
        document.getElementById("textInput").style.display = "block";
    } else {
        document.getElementById("textInput").style.display = "none";
    }
}

function analyzeEmotion() {
    let formData = new FormData();
    let analyzeButton = document.getElementById("analyzeButton");
    analyzeButton.disabled = true;

    if (mode === "text") {
        let text = document.getElementById("textInput").value;
        formData.append("text", text);
        fetch("/predict_text", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => displayResult(data))
            .finally(() => analyzeButton.disabled = false);

    } else if (mode === "speech") {
        analyzeButton.innerText = "🎤 Recording...";
        fetch("/predict_speech", { method: "POST" })
            .then(response => response.json())
            .then(data => displayResult(data))
            .finally(() => {
                analyzeButton.innerText = "Analyze Emotion";
                analyzeButton.disabled = false;
            });

    } else if (mode === "facial") {
        analyzeButton.innerText = "📷 Capturing...";
        fetch("/predict_facial", { method: "POST" })
            .then(response => response.json())
            .then(data => displayResult(data))
            .finally(() => {
                analyzeButton.innerText = "Analyze Emotion";
                analyzeButton.disabled = false;
            });
    }
}

function displayResult(data) {
    document.getElementById("result").innerText = "🎭 Detected Emotion: " + data.emotion;
    document.getElementById("suggestion").innerText = "💡 Suggested Activity: " + data.suggestion;
}
