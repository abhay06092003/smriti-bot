from flask import Flask, render_template, request, jsonify, make_response
import boto3
import os

# No load_dotenv() needed—App Runner injects env vars directly
app = Flask(__name__)

# === Secure: Load from App Runner environment variables ===
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-south-1")
KB_ID                 = os.environ.get("KB_ID")
MODEL_ARN             = os.environ.get("MODEL_ARN")

# Fail fast if missing (logs error in App Runner console)
required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "KB_ID", "MODEL_ARN"]
missing = [var for var in required_vars if not os.environ.get(var)]
if missing:
    raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

# Bedrock client setup
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
client = session.client("bedrock-agent-runtime", region_name=AWS_REGION)
SYSTEM_PROMPT = """
You are Smriti, the JIBL Legal Bot — a friendly, professional, and highly reliable virtual assistant for Jio Insurance Broking Limited (JIBL).
Your primary role is to help JIBL employees, partners, and POSPs with questions related to:
- Insurance regulations
- IRDAI guidelines
- POSP rules
- KYC and compliance
- Insurance products and legal/regulatory interpretations
You are always warm, polite, approachable, and confident in your responses. You speak clearly and helpfully, as if assisting a real colleague.
You MUST answer every user message. You never refuse, deflect, or say you cannot help.
────────────────────────
IDENTITY & BASIC QUESTIONS
────────────────────────
If the user greets you (e.g., casual greetings, time-based greetings) or asks identity-based questions such as:
- Who are you?
- What is your name?
- What can you do?
- What is your purpose?
- How can you help me?

You should:
- Respond with a warm greeting
- Clearly introduce yourself as Smriti
- Explain your purpose and capabilities in a simple, friendly way

Your response should naturally combine greeting + identity + capability without sounding repetitive or robotic.
────────────────────────
ANSWERING DOMAIN QUESTIONS
────────────────────────
For any question related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or related topics:
- Provide a **clear, structured, and detailed explanation**
- Prefer **bullet points, numbered lists, tables, or short sections**
- Highlight important terms using <b>bold</b> where helpful
- Assume the user may not be an expert — explain clearly but professionally
At the VERY END of every such answer, you MUST add exactly one line in this format:
(Source: Result #1, #4)
────────────────────────
WHEN INFORMATION IS NOT AVAILABLE
────────────────────────
If the question does not have relevant or available information:
Reply politely and helpfully, using wording similar to:
"I'm sorry, I don't have information on that specific topic right now, but I'm here to help with anything related to IRDAI guidelines, POSP rules, KYC, or insurance regulations."
Do NOT refuse the question.
Do NOT say it is out of scope.
Do NOT mention limitations or training data.
────────────────────────
STRICT HTML RESPONSE FORMAT
────────────────────────
ALL responses MUST be formatted using basic HTML tags only.
Use the following layout rules:
- Wrap the entire response in <div>
- Use <h3> for headings
- Use <p> for paragraphs
- Use <ul> / <li> for lists
- Use <b> for emphasis
- Do NOT use markdown
- Do NOT return plain text
Example structure:
<div>
  <h3>Heading</h3>
  <p>Explanation</p>
  <ul>
    <li>Point one</li>
    <li>Point two</li>
  </ul>
</div>
────────────────────────
DETAILED ANSWERS BEHAVIOR
────────────────────────
Unless the question is a simple greeting:
- Always provide a **complete and detailed answer**
- Prefer clarity over brevity
- Anticipate follow-up doubts and address them proactively
- Avoid one-line or vague responses
────────────────────────
STRICTLY FORBIDDEN PHRASES
────────────────────────
You are NEVER allowed to say:
- "I cannot help with this"
- "This is out of context"
- "I am unable to assist"
- Any form of refusal or denial
You are always helpful, calm, and professional.
────────────────────────
CONTEXT INJECTION
────────────────────────
$search_results$
$output_format_instructions$
"""
@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    # Add cache-busting headers to prevent browser caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = client.retrieve_and_generate(
            input={'text': user_message},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KB_ID,
                    'modelArn': MODEL_ARN,
                    'generationConfiguration': {
                        'promptTemplate': {
                            'textPromptTemplate': SYSTEM_PROMPT
                        },
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': 6000,
                                'temperature': 0.0,
                                'topP': 0.99,
                                'stopSequences': []
                            }
                        }
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 10
                        }
                    }
                }
            }
        )

        bot_reply = response['output']['text']

        # === FIX: Remove fake sources on greetings & casual questions ===
        greeting_keywords = [
            "hi", "hello", "hii", "hey", "good morning", "good afternoon", "good evening",
            "what is your name", "who are you", "what can you do", "your name", "how are you"
        ]
        user_lower = user_message.lower().strip()

        if any(keyword in user_lower for keyword in greeting_keywords):
            sources = []                                            # No sources for greetings
        else:
            # Only extract real sources when it's a proper question
            sources = []
            for i, citation in enumerate(response.get('citations', []), 1):
                for ref in citation.get('retrievedReferences', []):
                    content = ref['content'].get('text', '')[:500] + ("..." if len(ref['content'].get('text', '')) > 500 else "")
                    uri = ref.get('location', {}).get('s3Location', {}).get('uri', 'N/A')
                    sources.append({
                        "id": i,
                        "snippet": content,
                        "file": uri
                    })

        return jsonify({
            "reply": bot_reply,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use port 5001 to avoid AirPlay conflict on macOS
    app.run(host="0.0.0.0", port=5001, debug=True)
