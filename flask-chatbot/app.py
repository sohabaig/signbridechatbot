from flask import Flask, render_template, request, jsonify, session
import random

app = Flask(__name__)
# need secret key?

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    # again, trying to reset state
    session['quiz_state'] = {
        "score": 0,
        "completed_letters": [],
        "current_letter": random.choice(alphabet)
    }
    
    return render_template(
        "quiz.html",
        current_letter=session['quiz_state']["current_letter"],
        score=session['quiz_state']["score"]
    )

@app.route("/verify", methods=["POST"])
def verify():
    if 'quiz_state' not in session:
        # create new state
        session['quiz_state'] = {
            "score": 0,
            "completed_letters": [],
            "current_letter": random.choice(alphabet)
        }

    quiz_state = session['quiz_state']
    predicted_letter = quiz_state["current_letter"] 

    if predicted_letter == quiz_state["current_letter"]:
        quiz_state["score"] += 1
        quiz_state["completed_letters"].append(quiz_state["current_letter"])

        remaining_letters = list(set(alphabet) - set(quiz_state["completed_letters"]))
        if remaining_letters:
            quiz_state["current_letter"] = random.choice(remaining_letters)
            session['quiz_state'] = quiz_state 
            return jsonify({
                "success": True, 
                "current_letter": quiz_state["current_letter"], 
                "score": quiz_state["score"]
            })
        else:
            # reset score when quiz finishes
            final_score = quiz_state["score"]
            session['quiz_state'] = {  
                "score": 0,
                "completed_letters": [],
                "current_letter": random.choice(alphabet)
            }
            return jsonify({
                "success": True, 
                "completed": True, 
                "score": final_score,
                "current_letter": session['quiz_state']["current_letter"]
            })
    else:
        return jsonify({"success": False, "message": "Incorrect! Try again."})

if __name__ == "__main__":
    app.run(debug=True)