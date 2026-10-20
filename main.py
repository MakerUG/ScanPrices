import os
import json
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-5.4-mini"

RECEIPT_SCHEMA = {
    "name": "receipt_parse",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "merchant_name": {"type": ["string", "null"]},
            "total_amount": {"type": ["number", "null"]},
            "currency": {"type": ["string", "null"]},
            "transaction_date": {"type": ["string", "null"]},
            "receipt_type": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "price": {"type": ["number", "null"]},
                        "quantity": {"type": ["number", "null"]}
                    },
                    "required": ["name", "price", "quantity"]
                }
            }
        },
        "required": [
            "merchant_name",
            "total_amount",
            "currency",
            "transaction_date",
            "receipt_type",
            "category",
            "confidence",
            "items"
        ]
    }
}

TOTAL_ONLY_SCHEMA = {
    "name": "receipt_total_only",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "total_amount": {"type": ["number", "null"]},
            "currency": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"]}
        },
        "required": ["total_amount", "currency", "confidence"]
    }
}


@app.get("/health")
def health():
    return {"status": "ok"}


def call_structured_parse(image_base64: str, content_type: str, ocr_text: str):
    instructions = f"""
You extract receipt data for a shopping app.

Your task:
- Identify the merchant name.
- Return merchant_name as a clean human-readable store or company name.
- Do not return random OCR fragments as merchant name.
- If the merchant name is unclear, use the most likely business name visible on the receipt.
- For Ukrainian receipts, preserve the readable merchant name in Ukrainian if it is clearer than transliteration.
- Identify the final amount actually paid by the customer.
- Classify the receipt into exactly one spending category.
- Allowed categories:
  Groceries, Coffee, Restaurants, Pharmacy, Transport, Household, Personal Care, Electronics, Clothing, Entertainment, Services, Other.
- If unsure about category, use Other.
- Do NOT confuse the final total with authorization code, terminal code, invoice number,
  reference number, tax amount, subtotal, cashback, bonus balance, item code, phone number,
  date, time, or card suffix.
- Use the receipt image as the PRIMARY source.
- Use OCR text only as SECONDARY help.
- If OCR text looks corrupted, ignore it.
- If the amount is ambiguous, return null for that field and reduce confidence.
- Return only valid JSON matching the schema.

OCR text:
{ocr_text}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        text={
            "format": {
                "type": "json_schema",
                "name": RECEIPT_SCHEMA["name"],
                "schema": RECEIPT_SCHEMA["schema"],
                "strict": True
            }
        },
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions},
                    {
                        "type": "input_image",
                        "image_url": f"data:{content_type};base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return json.loads(response.output_text)


def call_total_only_parse(image_base64: str, content_type: str, ocr_text: str):
    instructions = f"""
Extract ONLY the final amount paid from this receipt.

Rules:
- Return the final paid total only.
- Ignore authorization code, invoice number, terminal code, reference number, tax amount,
  subtotal, cashback, bonus balance, item code, phone number, date, time, card suffix.
- Prefer values next to terms like:
  total, amount paid, amount, grand total, paid, sum, suma, do splaty.
- Use the receipt image as PRIMARY evidence.
- Use OCR text only if helpful.
- If uncertain, return null.

OCR text:
{ocr_text}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        text={
            "format": {
                "type": "json_schema",
                "name": TOTAL_ONLY_SCHEMA["name"],
                "schema": TOTAL_ONLY_SCHEMA["schema"],
                "strict": True
            }
        },
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions},
                    {
                        "type": "input_image",
                        "image_url": f"data:{content_type};base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return json.loads(response.output_text)


def needs_total_retry(parsed: dict) -> bool:
    total = parsed.get("total_amount")
    confidence = parsed.get("confidence")

    if total is None:
        return True

    if isinstance(total, (int, float)) and total <= 0:
        return True

    if confidence is None:
        return True

    if confidence < 0.75:
        return True

    return False


@app.post("/parse-receipt")
async def parse_receipt(
    image: UploadFile = File(...),
    ocr_text: str = Form("")
):
    image_bytes = await image.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    content_type = image.content_type or "image/jpeg"

    parsed = call_structured_parse(
        image_base64=image_base64,
        content_type=content_type,
        ocr_text=ocr_text
    )

    if needs_total_retry(parsed):
        total_retry = call_total_only_parse(
            image_base64=image_base64,
            content_type=content_type,
            ocr_text=ocr_text
        )

        if total_retry.get("total_amount") is not None:
            parsed["total_amount"] = total_retry.get("total_amount")
            parsed["currency"] = total_retry.get("currency") or parsed.get("currency")
            parsed["confidence"] = max(
                float(parsed.get("confidence") or 0),
                float(total_retry.get("confidence") or 0)
            )

    if not parsed.get("category"):
        parsed["category"] = "Other"

    return parsed


# Separate aggregate-only coaching endpoint; existing receipt parsing stays unchanged.
from spending_coach import create_spending_router
app.include_router(create_spending_router(client, MODEL_NAME))
