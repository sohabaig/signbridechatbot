window.onload = function () {
    const feedback = document.getElementById("feedback");
    const score = document.getElementById("score");
    const currentLetter = document.getElementById("current-letter");

    // trying to force reset score
    feedback.textContent = "";
    score.textContent = "Score: 0";
    
    console.log("Reset complete:", {
        feedback: feedback.textContent,
        score: score.textContent,
        currentLetter: currentLetter.textContent
    });
};

document.getElementById("submit-btn").addEventListener("click", async () => {
    const feedback = document.getElementById("feedback");
    const currentLetter = document.getElementById("current-letter");
    const score = document.getElementById("score");

    try {
        feedback.textContent = "";

        const response = await fetch("/verify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const result = await response.json();
        console.log("Server response:", result);

        if (result.success) {
            if (result.completed) {
                feedback.textContent = `Quiz Complete! Final Score: ${result.score}`;
                score.textContent = `Score: ${result.score}`;
                currentLetter.textContent = `Sign the letter: ${result.current_letter}`;
            } else {
                feedback.textContent = "Correct!";
                score.textContent = `Score: ${result.score}`;
                currentLetter.textContent = `Sign the letter: ${result.current_letter}`;
            }
        } else {
            feedback.textContent = "Incorrect, try again!";
        }
    } catch (error) {
        console.error("Error during verification:", error);
        feedback.textContent = "An error occurred. Please try again.";
    }
});