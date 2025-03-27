"""
deal_classifier.py

קובץ זה מכיל את הלוגיקה לניתוח הסנטימנט של תיאורי המודעות באמצעות Zero‑Shot Classification.
המטרה היא לסווג כל תיאור לאחת משלוש הקטגוריות:
    - "עסקה מצויינת: רכב במצב מצוין, התקבל בירושה, חייב למכור, עובר לחו\"ל"
    - "עסקה בינונית: רכב במצב רגיל, ללא סימנים לעסקה יוצאת דופן או בעיות"
    - "עסקה גרועה: רכב עם בעיות ותקלות, דורש תיקון כבד"

בנוסף, הפונקציות מחלצות את החלק הפשוט של התווית וממירות אותה לצבע מתאים.
"""

from transformers import pipeline

# הגדרת candidate_labels עם פירוט מלא לכל קטגוריה
candidate_labels = [
    "עסקה מצויינת: רכב במצב מצוין, התקבל בירושה, חייב למכור, עובר לחו\"ל",
    "עסקה בינונית: רכב במצב רגיל, ללא סימנים לעסקה יוצאת דופן או בעיות",
    "עסקה גרועה: רכב עם תקלות חמורות, בעיות מכניות קשות, דורש תיקונים יקרים, לא מומלץ"
]


# יצירת pipeline ל-zero-shot classification עם המודל facebook/bart-large-mnli
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

def classify_deal(description: str):
    """
    מקבלת תיאור מודעה ומסווגת אותו לפי הקטגוריות שהוגדרו.
    מחזירה:
       - full_label: התווית המלאה (כוללת את הפירוט)
       - score: הציון עבור אותה תווית
       - full_result: הפלט המלא מהמודל
    """
    result = classifier(description, candidate_labels)
    full_label = result["labels"][0]
    score = result["scores"][0]
    return full_label, score, result

def get_simple_label(full_label: str) -> str:
    """
    מפשטת את התווית – מחלצת את החלק לפני הקולון.
    לדוגמה, "עסקה מצויינת: רכב במצב מצוין, התקבל בירושה, חייב למכור, עובר לחו\"ל"
    תהפוך ל-"עסקה מצויינת".
    """
    if ":" in full_label:
        return full_label.split(":")[0].strip()
    return full_label.strip()

def get_color_for_deal(simple_label: str) -> str:
    """
    ממירה את התווית הפשוטה לצבע:
       - "עסקה מצויינת" -> "green"
       - "עסקה בינונית" -> "yellow"
       - "עסקה גרועה" -> "red"
       - אחרת -> "gray"
    """
    if simple_label == "עסקה מצויינת":
        return "green"
    elif simple_label == "עסקה בינונית":
        return "yellow"
    elif simple_label == "עסקה גרועה":
        return "red"
    else:
        return "gray"

# דוגמה לשימוש (ניתן להריץ את הקובץ הזה בנפרד):
if __name__ == "__main__":
    example_description = "רכב במצב מצוין, מחיר נמוך במיוחד, ירושתי – עסקה שלא תחזור!"
    full_label, score, result = classify_deal(example_description)
    simple_label = get_simple_label(full_label)
    color = get_color_for_deal(simple_label)
    print("Full label:", full_label)
    print("Simple label:", simple_label)
    print("Score:", score)
    print("Assigned color:", color)
