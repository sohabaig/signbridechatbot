async function pollPrediction() {
    try {
        const response = await fetch("/prediction");
        const result = await response.json();

        const predictedLetter = result.prediction || "None";
        const confidenceScore = result.confidence || 0.0;

        document.getElementById("predicted-letter").textContent = predictedLetter;
        document.getElementById("confidence-score").textContent = confidenceScore.toFixed(2);

        const currentLetterElement = document.getElementById("current-letter");
        const currentLetter = currentLetterElement.textContent.replace("Sign the letter: ", "");

        if (predictedLetter === currentLetter) {
            const feedback = document.getElementById("feedback");
            feedback.textContent = "Correct!";

            const verifyResponse = await fetch("/verify", { method: "POST" });
            const verifyResult = await verifyResponse.json();

            if (verifyResult.success) {
                currentLetterElement.textContent = `Sign the letter: ${verifyResult.current_letter}`;
                document.getElementById("score").textContent = `Score: ${verifyResult.score}`;
            }
        }
    } catch (error) {
        console.error("Error fetching prediction:", error);
    }
}

setInterval(pollPrediction, 1000);
