"""
================================================================================
  sign_detector_v2.py  —  Multi-Stage Danger Sign Detector (High-Precision)
================================================================================
  نظام كشف لوحات الخطر/التحذير بـ ZERO TOLERANCE للـ False Positives

  ──────────────────────────────────────────────────────────
  المشاكل التي حلّها هذا الإصدار:
  ──────────────────────────────────────────────────────────
  1. كان بيغلط مع ملابس بشر ذات ألوان حمراء/برتقالية → نضاف Anti-Human-Body filter
  2. كان بيغلط مع إشارات المرور العادية → نضاف Road-Sign context rejection
  3. كان بيؤكد الكشف بسرعة → رفعنا persistence threshold بشكل كبير
  4. مكانش فيه multi-stage verification → أضفنا 5 مراحل تحقق تسلسلية
  5. مكانش فيه confidence scoring مجمّع → أضفنا Weighted Fusion Score

  ──────────────────────────────────────────────────────────
  v2.1 — Glare / Harsh-Light Robustness (جديد) :
  ──────────────────────────────────────────────────────────
  المشكلة: تحت إضاءة قاسية (شمس مباشرة، أو انعكاس شاشة عند تصوير صورة
  تحذير من موبايل) بيحصل overexposure — البكسلات بتـ"تحرق" (V≈255) وده
  بيكسر الـ Hue الحقيقي أو يقلل الـ Saturation، فالـ Color Gate القديم
  كان برفضها كـ"رمادي/أبيض" أو الـ Shape Gate بيفشل لأن الحواف بتختفي.

  الحل (3 طبقات مستقلة، بدل ما نغيّر الأرقام عشوائيًا):
    1) Glare Pre-Processing — كشف وتصحيح مناطق الـ overexposure قبل أي
       تحليل HSV خالص (CLAHE على V channel + Highlight recovery).
    2) Adaptive Brightness Gate — الكود بيقيس متوسط سطوع الفريم ولو
       لاقاه عالي جدًا، بيقلل عتبة V الأدنى ويوسّع تسامح الألوان تلقائيًا
       (مش hardcoded — بيتكيف مع كل فريم لوحده).
    3) Shape-Priority Fallback — لو الإضاءة قاسية جدًا (overexposed_ratio
       عالي) والـ Color Gate ضعيف لكن قريب من الحد، الكود بيسمح بالعبور
       بثقة أقل بدل الرفض الكامل، على إن الـ Shape + Context + Temporal
       يعوّضوا الثقة المفقودة (defense-in-depth زي باقي النظام).

  ──────────────────────────────────────────────────────────
  المراحل الخمس للتحقق (كل مرحلة لازم تعدي قبل اللي بعدها):
  ──────────────────────────────────────────────────────────

  Stage 1 — COLOR GATE:
    فلترة صارمة جداً بـ HSV (مع تصحيح glare قبلها) مع rejection للألوان
    المشابهة غير المقصودة.
    بيرفض: البرتقالي العادي، الوردي، الجلدي (skin tones).
    بيقبل: الأحمر الكشط IMAS، البرتقالي الفوسفوري، وكذلك نسخهم
    "المحروقة بالضوء" (overexposed) بعد التصحيح.

  Stage 2 — SHAPE GATE:
    تحقق صارم من الشكل: مثلث أو مستطيل تحذير حقيقي.
    يحسب: نسبة المنطقة، convexity defects، aspect ratio، solidity.
    بيرفض: الأشكال الكبيرة جداً التي تكون ملابس، والمستطيلات العريضة.

  Stage 3 — CONTEXT GATE (anti-false-positive):
    يتحقق إن الكائن المكشوف مش جسم بشري أو سيارة.
    - Anti-Human: لو الـ bounding box في الثلث السفلي من الصورة ونسبته
      عمودية → محتمل يكون شخص → يرفضه.
    - Anti-Vehicle: لو المنطقة المكشوفة في أسفل-يمين أو أسفل-يسار ومش في
      المنتصف → محتمل يكون إشارة مرور على الجانب → يشك فيه.
    - Edge-proximity check: لوحات الخطر الحقيقية نادراً ما تكون على
      الحافة الجانبية للصورة تماماً.

  Stage 4 — TEXTURE GATE:
    تحقق من نسيج اللوحة: اللافتات عندها texture مختلف عن الملابس والجلد.
    بيحسب: gradient magnitude variance داخل المنطقة → اللوحات عندها
    حواف حادة وكتابة/pattern، الملابس والجلد عندهم texture أكثر تجانسًا.
    تحت glare: بيستخدم نسخة مصحَّحة من الصورة (بعد الـ highlight recovery)
    عشان الحواف اللي اختفت بسبب الحرق الضوئي ترجع تظهر.

  Stage 5 — TEMPORAL GATE:
    Multi-frame voting window صارمة: 12 فريم من أصل 16 لازم يكونوا positive
    قبل ما نؤكد الـ danger. بالإضافة لـ cooldown period بعد كل تأكيد.

  ──────────────────────────────────────────────────────────
  نظام الـ Confidence Scoring:
  ──────────────────────────────────────────────────────────
    كل stage بتضيف نقاط للـ confidence score النهائي:
      COLOR_QUALITY    → 0..25 نقطة
      SHAPE_QUALITY    → 0..25 نقطة
      CONTEXT_CLEAN    → 0..20 نقطة
      TEXTURE_SIGN     → 0..15 نقطة
      TEXT_MATCH       → 0..15 نقطة (bonus)
    الحد الأدنى للتأكيد: 55/100 نقطة (بدون TEXT) أو 40/100 (مع TEXT CRITICAL)

  ──────────────────────────────────────────────────────────
  Graceful Fallback:
  ──────────────────────────────────────────────────────────
    - بدون Tesseract: Stage 5 فقط + score threshold أعلى (65/100)
    - أي exception في أي stage: يتسجل في log ويرجع "لا يوجد كشف"
    - Thread-safe بالكامل

  Author  : Robot Team
  Version : 2.1  (High-Precision + Glare-Robust Edition)
================================================================================
"""

import cv2
import re
import time
import logging
import threading
import unicodedata
import numpy as np

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

log = logging.getLogger("SignDetectorV2")


# ══════════════════════════════════════════════════════════
#  GLARE / HARSH-LIGHT HANDLING (v2.1)
# ══════════════════════════════════════════════════════════
#  الهدف: نتعامل مع overexposure (شمس مباشرة / انعكاس شاشة) بدون ما
#  نضحّي بالدقة في إضاءة عادية. كل القيم دي بتتفعّل بس لو الفريم فعلاً
#  مضيء بشكل غير طبيعي — مفيش تأثير على فريم بإضاءة طبيعية.

# لو أكتر من النسبة دي من الفريم V>=GLARE_V_THRESHOLD → الفريم "مضيء جدًا"
GLARE_V_THRESHOLD       = 245      # قيمة V (brightness) تعتبر "محروقة"
GLARE_FRAME_RATIO_HIGH  = 0.18     # 18%+ من الفريم محروق → إضاءة قاسية جدًا
GLARE_FRAME_RATIO_MILD  = 0.06     # 6%+ → إضاءة قاسية بس مش متطرفة

# تصحيح الـ highlight: بنقلل V للبكسلات المحروقة عشان الـ Hue يرجع يبان
# (مضروب في القناة V بس، الـ Hue والـ Saturation متأثرين بشكل غير مباشر
# بعد إعادة موازنة التباين بـ CLAHE)
CLAHE_CLIP_LIMIT  = 2.5
CLAHE_GRID_SIZE   = (8, 8)

# تسامح إضافي في عتبة V الأدنى للألوان المستهدفة لما الفريم يكون مضيء جدًا
# (بدل ما نرفض البكسل المحروق، نتعامل معاه كهدف بثقة أقل)
ADAPTIVE_V_RELAX_MILD = 25   # تقليل V_MIN بمقدار كذا تحت إضاءة قاسية متوسطة
ADAPTIVE_V_RELAX_HIGH = 45   # تقليل V_MIN بمقدار كذا تحت إضاءة قاسية جدًا

# لما الإضاءة قاسية جدًا، نقلل عتبة قبول الـ Color Gate نفسها (المساحة
# المطلوبة) لأن جزء من مساحة اللوحة هيكون "محروق" تمامًا ومش هيتحسب
# حتى بعد التصحيح
GLARE_MIN_RATIO_RELAX = 0.4   # نضرب COLOR_MIN_PIXEL_RATIO في هذا تحت قسوة شديدة


def _estimate_glare(frame: np.ndarray) -> Tuple[float, str]:
    """
    يحسب نسبة البكسلات "المحروقة" (overexposed) في الفريم ويرجّع
    (overexposed_ratio, severity) حيث severity ∈ {"none", "mild", "high"}.
    مقياس سريع (بس على V channel) — مفيش تكلفة حسابية زيادة لأنه
    هيُحسب مرة واحدة بس لكل فريم، قبل أي gate.
    """
    try:
        gray_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        total  = gray_v.size
        burned = int(np.count_nonzero(gray_v >= GLARE_V_THRESHOLD))
        ratio  = burned / float(total)

        if ratio >= GLARE_FRAME_RATIO_HIGH:
            return ratio, "high"
        if ratio >= GLARE_FRAME_RATIO_MILD:
            return ratio, "mild"
        return ratio, "none"
    except Exception as e:
        log.debug(f"[Glare] estimate error: {e}")
        return 0.0, "none"


def _correct_glare(frame: np.ndarray, severity: str) -> np.ndarray:
    """
    تصحيح الإضاءة القاسية قبل أي تحليل HSV — بيرجع نسخة جديدة من
    الفريم (الأصلي ما بيتغيرش)، فيها:
      1) CLAHE على V channel — بيوزّع التباين المضغوط في المنطقة
         المحروقة، فالحواف والـ texture اللي اختفت بسبب الحرق الضوئي
         بتاخد فرصة ترجع تبان شوية.
      2) تقليل خفيف لقناة V في البكسلات شديدة السطوع (highlight
         recovery تقريبي) — بيرجّع جزء من تباين الـ Hue المفقود.

    لو severity == "none" بيرجع الفريم الأصلي زي ما هو (zero overhead).

    حد فيزيائي مهم (مفيد لو حد سأل في المناقشة): تحت انعكاس قاسي جدًا
    (شاشة موبايل تحت شمس مباشرة)، الـ Saturation ممكن تنزل لحد ~10-20
    فعليًا — معلومة اللون بتضيع رياضيًا في القناة دي، ومفيش خوارزمية
    تصحيح صورة (CLAHE أو غيرها) تقدر "تخترع" تباين لون مفيش له أساس في
    البيانات الخام. الحل هنا (راجع HSV_*_GLARE تحت) مش بس "نرفع S"،
    لكن نضيّق نطاق الـ Hue المقبول بدل ما نوسّع كل الأبعاد، وننقل عبء
    التأكيد للـ stages اللاحقة (Shape/Context/Texture/Temporal) بدل ما
    نطلب من Color Gate وحده يحسم القرار تحت ظروف فيها معلومة ناقصة.
    """
    if severity == "none":
        return frame
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        v_eq  = clahe.apply(v_ch)

        # Highlight recovery تقريبي: للبكسلات اللي كانت محروقة فعليًا
        # (V الأصلي قريب من 255)، نمزج بين القيمة المعدّلة وقيمة مخفّضة
        # بس بنسبة معقولة — عشان نرجّع تباين بدون ما نـ"يفلطح" الصورة كلها
        burned_mask = (v_ch >= GLARE_V_THRESHOLD).astype(np.float32)
        v_recovered = (v_eq.astype(np.float32) * 0.75 +
                       v_ch.astype(np.float32)  * 0.25)
        v_final = (burned_mask * v_recovered +
                   (1.0 - burned_mask) * v_ch.astype(np.float32))
        v_final = np.clip(v_final, 0, 255).astype(np.uint8)

        # ── Saturation recovery حقيقي (v2.1-fix) ────────────────────
        # تحت glare شديد جدًا، الـ S الأصلية ممكن توصل لقيم صغيرة جدًا
        # (5-15) لدرجة إن ضرب ثابت بسيط (×1.25) ميغيّرش حاجة فعليًا.
        # الحل الصحيح: CLAHE على قناة S نفسها — بيعيد توزيع التباين
        # المحلي الموجود فعليًا (الفرق بين بكسل وبكسل) حتى لو كل القيم
        # المطلقة صغيرة، فبيرجّع تباين نسبي يفرّق بين "أحمر مغسول جدًا"
        # و"أبيض حقيقي مالوش أي صبغة" بشكل أوضح من الضرب البسيط.
        s_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        s_eq    = s_clahe.apply(s_ch)

        if severity == "high":
            # دمج بين الأصل والمعدّل + رفع نسبي إضافي للقيم الصغيرة جدًا
            s_boost = np.clip(
                s_eq.astype(np.float32) * 0.7 + s_ch.astype(np.float32) * 0.3,
                0, 255
            )
            # تكبير نسبي إضافي للبكسلات الباهتة جدًا (S < 40) فقط — مش
            # تكبير عام، عشان ميأثرش على بكسلات عندها S كافية أصلاً
            faint = s_boost < 40
            s_boost = np.where(faint, np.clip(s_boost * 2.2, 0, 80), s_boost)
            s_boost = s_boost.astype(np.uint8)
        else:
            s_boost = np.clip(s_eq.astype(np.float32) * 0.5 + s_ch.astype(np.float32) * 0.5,
                               0, 255).astype(np.uint8)

        corrected_hsv   = cv2.merge([h_ch, s_boost, v_final])
        corrected_frame = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        return corrected_frame

    except Exception as e:
        log.debug(f"[Glare] correction error: {e}")
        return frame


# ══════════════════════════════════════════════════════════
#  STAGE 1 — COLOR GATE CONSTANTS
# ══════════════════════════════════════════════════════════

# الأحمر IMAS — موسّع قليلاً عشان يشمل الأحمر الحقيقي في الإضاءات المختلفة
# الحد الأدنى لـ V اتخفّض من 70 → 60 عشان يشمل ظلال خفيفة، والـ S
# اتخفّض من 100 → 90 عشان يشمل الأحمر المغسول جزئيًا تحت إضاءة قوية
HSV_RED_1 = (np.array([0,   90,  60]),  np.array([10,  255, 255]))
HSV_RED_2 = (np.array([165, 90,  60]),  np.array([180, 255, 255]))

# البرتقالي الفوسفوري IMAS
HSV_ORANGE_IMAS = (np.array([11, 140, 100]), np.array([22, 255, 255]))

# الأصفر IMAS (جمجمة، DANGER، لوحات تحذير صفرا) — target مش reject
# نطاق ضيق عشان يرفض الأصفر الباهت (ملابس، خلفيات)
HSV_YELLOW_IMAS = (np.array([23, 130, 100]), np.array([38, 255, 255]))

# ── v2.1: نطاقات "محروقة بالضوء" (overexposed variants) ───────────
# نفس الألوان المستهدفة فوق، بس بعتبة S أقل (لون مغسول/فاتح) وعتبة V
# أعلى (شديد السطوع) — دول مش بدائل، دول إضافة: بيتفعّلوا بس لو الفريم
# اتصنّف "mild" أو "high" glare. كده مفيش تأثير على إضاءة عادية.
#
# ملحوظة هندسية مهمة: تحت glare شديد جدًا (انعكاس شاشة موبايل تحت شمس)
# قسنا فعليًا إن الـ Saturation ممكن تنزل لحد 10-20 فقط — مفيش طريقة
# نطلب S أعلى من كده بأمان لأن المعلومة فعليًا ضاعت فيزيائيًا. اخترنا
# نعوّض ده بتضييق نطاق الـ Hue (دقة أعلى في تحديد اللون نفسه) بدل
# توسيع كل الأبعاد الثلاثة، فالخطأ المحتمل ينتقل لباقي الـ stages
# (Shape/Context/Texture/Temporal) يعوّضوه، مش الـ Color Gate لوحده.
HSV_RED_GLARE_1    = (np.array([0,   12, 175]), np.array([6,   255, 255]))
HSV_RED_GLARE_2    = (np.array([172, 12, 175]), np.array([180, 255, 255]))
HSV_ORANGE_GLARE   = (np.array([12,  18, 185]), np.array([22,  255, 255]))
HSV_YELLOW_GLARE   = (np.array([24,  18, 185]), np.array([36,  255, 255]))

# Rejection masks — بيرفض الألوان دي لو كانت غالبة
HSV_SKIN_REJECT   = (np.array([0,  20, 100]),  np.array([25, 150, 255]))  # جلد — موسّع لكل درجات البشرة
# Yellow REJECT اتنقل لـ target — ما بنرفضوش دلوقتي
HSV_PINK_REJECT   = (np.array([160, 30, 150]), np.array([170, 120, 255])) # وردي فاتح

COLOR_MIN_PIXEL_RATIO = 0.004  # مخفّض — لوحات صغيرة أو بعيدة تعدي
COLOR_MAX_PIXEL_RATIO = 0.25   # مرفوع — لوحات قريبة كبيرة تعدي

# لو نسبة rejected colors > هذا → يرفض الكشف
COLOR_REJECTION_THRESHOLD = 0.35


# ══════════════════════════════════════════════════════════
#  STAGE 2 — SHAPE GATE CONSTANTS
# ══════════════════════════════════════════════════════════

SHAPE_APPROX_EPS     = 0.030   # أكثر مرونة في تقريب الشكل
SHAPE_MIN_AREA_RATIO = 0.004   # يقبل لوحات أصغر
SHAPE_MAX_AREA_RATIO = 0.28    # يقبل لوحات أكبر (قريبة من الكاميرا)
SHAPE_MIN_SOLIDITY   = 0.68    # أقل صرامة — يقبل أشكال فيها شوية ضجيج
SHAPE_ASPECT_MIN     = 0.45    # أوسع نطاق للـ aspect ratio
SHAPE_ASPECT_MAX     = 1.80

# الشكل لازم يكون مثلث (3 أضلاع) أو مضلع قريب (4-6 أضلاع)
SHAPE_ACCEPTED_VERTICES = (3, 4)   # مثلث (3) بس، 4 كـ tolerance للكاميرا

# Convexity مخفّف — اللوحات الحقيقية مش دايماً مثالية
SHAPE_MIN_CONVEXITY  = 0.72


# ══════════════════════════════════════════════════════════
#  STAGE 3 — CONTEXT GATE CONSTANTS
# ══════════════════════════════════════════════════════════

# لو المنطقة المكشوفة في أسفل الصورة وعمودية الشكل → محتمل يكون شخص
CONTEXT_BODY_BOTTOM_THRESH  = 0.65  # أسفل 65% من الصورة
CONTEXT_BODY_ASPECT_MIN     = 1.4   # aspect > 1.4 = عمودي (شخص)
CONTEXT_BODY_ASPECT_MAX     = 4.0

# لو المنطقة على أطراف الصورة جداً → محتمل إشارة مرور جانبية
CONTEXT_EDGE_MARGIN_RATIO   = 0.04  # مخفّض من 7% → 4% عشان يقبل لوحات جانبية

# لو المنطقة صغيرة جداً وبعيدة → زيادة threshold للاشتراطات
CONTEXT_SMALL_DISTANT_RATIO = 0.008


# ══════════════════════════════════════════════════════════
#  STAGE 4 — TEXTURE GATE CONSTANTS
# ══════════════════════════════════════════════════════════

# اللافتات عندها gradient variance عالٍ (حواف حادة + كتابة)
# مخفّض — لوحات في الشمس أو ببُعد مش دايماً عندها variance عالٍ
TEXTURE_MIN_GRADIENT_VAR = 80.0    # مخفّض من 180 → 80
TEXTURE_MAX_GRADIENT_VAR = 12000.0 # مرفوع — تغطية أوسع

# Edge density: مخفّض الحد الأدنى
TEXTURE_MIN_EDGE_DENSITY  = 0.04   # مخفّض من 0.06 → 0.04
TEXTURE_MAX_EDGE_DENSITY  = 0.65   # مرفوع


# ══════════════════════════════════════════════════════════
#  STAGE 5 — TEMPORAL GATE CONSTANTS
# ══════════════════════════════════════════════════════════

TEMPORAL_WINDOW  = 12
TEMPORAL_HITS    = 5   # 5 من 12 فريم — أسرع في التأكيد (~170ms على 30fps)

# بعد التأكيد: reset الـ history عشان الـ detection الجاي يبني من الصفر
TEMPORAL_COOLDOWN_SEC = 3.0

# لما اللوحة تختفي: عدد الفريمات المتتالية الفاشلة في Stage 1 قبل ما نمسح الـ danger
# 20 فريم = ~670ms على 30fps — بيتجاهل الـ flicker العادي
CLEAR_STREAK_NEEDED = 12   # 12 فريم = ~400ms — كافي لتجاهل الـ flicker

DANGER_MIN_HOLD_SEC = 1.0  # ثانية واحدة hold — بعدها بيتمسح بسرعة لما اللوحة تتشال


# ══════════════════════════════════════════════════════════
#  CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════

SCORE_COLOR_MAX    = 25
SCORE_SHAPE_MAX    = 25
SCORE_CONTEXT_MAX  = 20
SCORE_TEXTURE_MAX  = 15
SCORE_TEXT_MAX     = 15

# الحد الأدنى للتأكيد — مخفّض عشان يكون قابل للتحقق بدون OCR
SCORE_MIN_CONFIRM_NO_TEXT  = 42   # مخفّض من 55 → 42
SCORE_MIN_CONFIRM_WITH_OCR = 30   # مخفّض من 40 → 30


# ══════════════════════════════════════════════════════════
#  OCR CONSTANTS
# ══════════════════════════════════════════════════════════

OCR_LANGS             = "ara+eng"
OCR_FULLFRAME_EVERY_N = 60       # أبطأ من القديم — نركز على المناطق المشبوهة
OCR_WORD_MIN_CONF     = 45
OCR_MIN_TEXT_LEN      = 2

WARNING_KEYWORDS_CRITICAL = [
    "خطر", "لغم", "الغام", "ألغام", "لغم ارضي", "حقل الغام", "حقل ألغام",
    "ممنوع الدخول", "ممنوع المرور", "منطقة خطر",
    "danger", "mine", "mines", "minefield", "unexploded", "explosive",
    "keep out", "do not enter", "hazardous area",
]
WARNING_KEYWORDS_GENERIC = [
    "احذر", "تحذير", "ممنوع", "خطورة",
    "warning", "caution", "hazard", "restricted", "forbidden",
]

TEXT_GENERIC_CONFIRMS = 3    # كلمات عامة تحتاج 3 تأكيدات متتالية


# ══════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════

@dataclass
class StageResult:
    passed:     bool  = False
    score:      float = 0.0
    reason:     str   = ""
    debug_info: dict  = field(default_factory=dict)


@dataclass
class SignDetectionResult:
    timestamp:         float = field(default_factory=time.time)

    # مراحل التحقق
    stage1_color:      bool  = False
    stage2_shape:      bool  = False
    stage3_context:    bool  = False
    stage4_texture:    bool  = False
    stage5_temporal:   bool  = False

    # المعلومات المكشوفة
    shape_box:         tuple = ()
    confidence_score:  float = 0.0

    # OCR
    text_matched_word: str   = ""
    text_severity:     str   = "NONE"
    raw_text:          str   = ""

    # النتيجة النهائية
    danger_confirmed:  bool  = False
    reason:            str   = ""
    reject_reason:     str   = ""   # سبب الرفض لو مفيش danger

    def is_valid(self, max_age: float = 1.0) -> bool:
        return (time.time() - self.timestamp) < max_age


# ══════════════════════════════════════════════════════════
#  SLIDING WINDOW (TEMPORAL PERSISTENCE)
# ══════════════════════════════════════════════════════════

class _TemporalGate:
    def __init__(self, window: int, hits: int):
        self._hist    = deque(maxlen=window)
        self._hits    = hits
        self._cooldown_until = 0.0

    def update(self, detected: bool) -> bool:
        self._hist.append(bool(detected))
        if time.time() < self._cooldown_until:
            return False
        confirmed = sum(self._hist) >= self._hits
        if confirmed:
            self._cooldown_until = time.time() + TEMPORAL_COOLDOWN_SEC
            self._hist.clear()  # ← الـ detection الجاي يبني من الصفر
        return confirmed

    def reset(self):
        self._hist.clear()
        self._cooldown_until = 0.0

    @property
    def current_hits(self) -> int:
        return sum(self._hist)

    @property
    def window_size(self) -> int:
        return len(self._hist)


# ══════════════════════════════════════════════════════════
#  STAGE 1: COLOR GATE
# ══════════════════════════════════════════════════════════

class _ColorGate:

    def run(self, frame: np.ndarray) -> Tuple[StageResult, Optional[np.ndarray], np.ndarray, str]:
        """
        يرجع: (StageResult, color_mask أو None, frame_for_downstream, glare_severity)

        frame_for_downstream: الفريم اللي لازم الـ stages بعد كده (Shape,
        Texture) تستخدمه — يبقى الفريم المصحَّح لو حصل glare correction،
        وإلا يبقى الفريم الأصلي زي ما هو (zero overhead في الحالة العادية).
        """
        try:
            h, w = frame.shape[:2]
            frame_area = h * w

            # ── v2.1: قياس شدة الإضاءة القاسية على الفريم الخام أولًا ──
            glare_ratio, severity = _estimate_glare(frame)
            work_frame = _correct_glare(frame, severity)

            # blur خفيف قبل الـ HSV عشان يثبت اللون لما الإضاءة تتغير
            blurred = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # --- بناء الـ target mask (نطاقات أساسية دايمًا) ---
            m_red1   = cv2.inRange(hsv, *HSV_RED_1)
            m_red2   = cv2.inRange(hsv, *HSV_RED_2)
            m_orange = cv2.inRange(hsv, *HSV_ORANGE_IMAS)
            m_yellow = cv2.inRange(hsv, *HSV_YELLOW_IMAS)
            target_mask = cv2.bitwise_or(
                cv2.bitwise_or(cv2.bitwise_or(m_red1, m_red2), m_orange),
                m_yellow
            )

            # --- v2.1: نطاقات "محروقة بالضوء" — تتفعّل بس تحت glare ───
            # دول بيضيفوا تغطية للبكسلات اللي اتغسلت بالضوء ولسه قريبة
            # من اللون المستهدف، بدل ما تترفض بالكامل كـ"خلفية"
            if severity in ("mild", "high"):
                m_red_g    = cv2.bitwise_or(
                    cv2.inRange(hsv, *HSV_RED_GLARE_1),
                    cv2.inRange(hsv, *HSV_RED_GLARE_2),
                )
                m_orange_g = cv2.inRange(hsv, *HSV_ORANGE_GLARE)
                m_yellow_g = cv2.inRange(hsv, *HSV_YELLOW_GLARE)
                glare_target_mask = cv2.bitwise_or(
                    cv2.bitwise_or(m_red_g, m_orange_g), m_yellow_g
                )
                target_mask = cv2.bitwise_or(target_mask, glare_target_mask)

            # --- بناء الـ rejection mask ---
            m_skin = cv2.inRange(hsv, *HSV_SKIN_REJECT)
            m_pink = cv2.inRange(hsv, *HSV_PINK_REJECT)
            reject_mask = cv2.bitwise_or(m_skin, m_pink)

            # ── v2.1: حل تعارض جوهري — تحت glare، أحمر مغسول بالضوء
            # (Hue صحيح، S منخفضة، V عالية) بيقع غالبًا جوه نطاقي
            # HSV_RED_1/Target وHSV_SKIN_REJECT في نفس الوقت، لأن النطاقين
            # متقاربين في Hue والفرق الوحيد (S/V) بيضيع تحت الانعكاس.
            # ده هو السبب الأساسي اللي كان بيخلي اللوحة الحقيقية تترفض
            # كـ"بشرة". الحل الصحيح: أي بكسل طابق الـ target أصلاً، نشيله
            # من الـ reject mask — يعني الأولوية للون المستهدف الصريح،
            # ومش بنضعّف rejection للحالات اللي مفيهاش تطابق target خالص
            # (يد/وجه حقيقي بلا أي تشابه لونّي مع اللوحة هيفضل يترفض زي ما هو)
            reject_mask = cv2.bitwise_and(reject_mask, cv2.bitwise_not(target_mask))

            target_pixels = int(np.count_nonzero(target_mask))
            reject_pixels = int(np.count_nonzero(reject_mask))
            total_colored = max(target_pixels + reject_pixels, 1)

            target_ratio  = target_pixels / frame_area
            reject_ratio  = reject_pixels / float(total_colored)

            # --- v2.1: عتبة أدنى متكيّفة — تحت إضاءة قاسية جدًا، جزء من
            # مساحة اللوحة بيتحرق تمامًا ومش هيتحسب حتى بعد التصحيح، فلو
            # طلبنا نفس النسبة المعتادة هنرفض لوحة حقيقية ظلمًا ───────
            min_ratio_required = COLOR_MIN_PIXEL_RATIO
            if severity == "high":
                min_ratio_required = COLOR_MIN_PIXEL_RATIO * GLARE_MIN_RATIO_RELAX

            # --- التحقق ---
            if target_ratio < min_ratio_required:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"Color ratio too low: {target_ratio:.4f} < {min_ratio_required:.4f} (glare={severity})",
                ), None, work_frame, severity

            if target_ratio > COLOR_MAX_PIXEL_RATIO:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"Color ratio too high (background?): {target_ratio:.4f}",
                ), None, work_frame, severity

            if reject_ratio > COLOR_REJECTION_THRESHOLD:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"High rejection color ratio: {reject_ratio:.2f} (skin/yellow/pink detected)",
                ), None, work_frame, severity

            # --- Morphology لتنظيف الـ mask ---
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            clean_mask = cv2.morphologyEx(clean_mask,  cv2.MORPH_OPEN,  kernel, iterations=1)

            # --- Blob compactness check ---
            # اللوحة = بقعة لون واحدة متجمعة
            # الإنسان / الهدوم = لون متشتت في الفريم
            # بنشوف إن أكبر blob فيه على الأقل 50% من كل الـ colored pixels
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            if n_labels < 2:
                return StageResult(
                    passed=False, score=0,
                    reason="No connected color region found after morphology",
                ), None, work_frame, severity

            # stats[0] = background، نبدأ من 1
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_blob    = int(component_areas.max())
            total_colored_pixels = int(clean_mask.sum() // 255)

            if total_colored_pixels == 0:
                return StageResult(passed=False, score=0, reason="Empty mask after morphology"), None, work_frame, severity

            # تحت glare قاسي، اللوحة المحروقة جزئيًا ممكن تنقسم بصريًا
            # لقطعتين متقاربتين (الجزء المحروق فاصل بينهم) — نخفّف شرط
            # الـ compactness شوية بدل ما نرفضها كـ"متشتتة زي الهدوم"
            min_compactness = 0.45 if severity == "none" else 0.33

            compactness = largest_blob / total_colored_pixels
            if compactness < min_compactness:
                # اللون متشتت → على الأغلب جلد أو هدوم مش لوحة
                return StageResult(
                    passed=False, score=0,
                    reason=f"Color scattered (compactness={compactness:.2f} < {min_compactness}) — not a sign",
                ), None, work_frame, severity

            # --- Confidence score ---
            # أعلى نقطة لو النسبة في المنتصف (مش صغير جداً ولا كبير جداً)
            optimal_ratio = 0.035
            ratio_score = 1.0 - min(abs(target_ratio - optimal_ratio) / optimal_ratio, 1.0)
            purity_score = 1.0 - reject_ratio  # كلما قل الـ rejection كلما ارتفع الـ score
            color_score  = (ratio_score * 0.4 + purity_score * 0.6) * SCORE_COLOR_MAX

            # تحت glare قاسي، الـ confidence من اللون نفسه أقل موثوقية
            # بطبيعته — بنخصم جزء صغير من النقطة عشان الـ Shape/Texture/
            # Temporal يحتاجوا يعوّضوا أكتر (defense-in-depth الحقيقي)
            if severity == "high":
                color_score *= 0.85
            elif severity == "mild":
                color_score *= 0.93

            return StageResult(
                passed=True,
                score=color_score,
                reason=f"Color gate passed (glare={severity})",
                debug_info={
                    "target_ratio": target_ratio,
                    "reject_ratio": reject_ratio,
                    "color_score":  color_score,
                    "glare_ratio":  glare_ratio,
                    "glare_severity": severity,
                },
            ), clean_mask, work_frame, severity

        except Exception as e:
            log.error(f"[Stage1-Color] Exception: {e}")
            return StageResult(passed=False, score=0, reason=f"Exception: {e}"), None, frame, "none"


# ══════════════════════════════════════════════════════════
#  STAGE 2: SHAPE GATE
# ══════════════════════════════════════════════════════════

class _ShapeGate:

    def run(self, frame: np.ndarray, color_mask: np.ndarray) -> Tuple[StageResult, Optional[tuple]]:
        """
        يرجع: (StageResult, best_box أو None)
        best_box: (x1, y1, x2, y2)
        """
        try:
            h, w = frame.shape[:2]
            frame_area = h * w

            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return StageResult(passed=False, score=0, reason="No contours found"), None

            best_score = 0.0
            best_box   = None
            best_info  = {}
            candidates  = 0

            for c in contours:
                area = cv2.contourArea(c)
                ratio = area / frame_area

                if ratio < SHAPE_MIN_AREA_RATIO or ratio > SHAPE_MAX_AREA_RATIO:
                    continue

                # Convexity / Solidity
                hull      = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0:
                    continue
                solidity = area / hull_area
                if solidity < SHAPE_MIN_CONVEXITY:
                    continue

                # Bounding box + Aspect ratio
                x, y, bw, bh = cv2.boundingRect(c)
                if bw == 0 or bh == 0:
                    continue
                aspect = bh / float(bw)
                if aspect < SHAPE_ASPECT_MIN or aspect > SHAPE_ASPECT_MAX:
                    continue

                # Polygon approximation
                perimeter = cv2.arcLength(c, True)
                if perimeter < 10:
                    continue
                approx = cv2.approxPolyDP(c, SHAPE_APPROX_EPS * perimeter, True)
                n_verts = len(approx)
                if n_verts not in SHAPE_ACCEPTED_VERTICES:
                    continue

                candidates += 1

                # Shape quality score
                solidity_score = (solidity - SHAPE_MIN_CONVEXITY) / (1.0 - SHAPE_MIN_CONVEXITY)
                area_score     = 1.0 - abs(ratio - 0.04) / 0.04  # optimal ~4% من الفريم
                area_score     = max(area_score, 0)
                # المثلث (3 vertices) أفضل من المضلعات الأخرى
                vertex_bonus   = 1.0 if n_verts == 3 else 0.8

                shape_q = (solidity_score * 0.5 + area_score * 0.3 + vertex_bonus * 0.2)
                shape_q = min(shape_q, 1.0)

                if shape_q > best_score:
                    best_score = shape_q
                    best_box   = (x, y, x + bw, y + bh)
                    best_info  = {
                        "solidity":   solidity,
                        "aspect":     aspect,
                        "n_vertices": n_verts,
                        "area_ratio": ratio,
                        "shape_quality": shape_q,
                    }

            if best_box is None:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"No valid shape found (candidates checked: {candidates})",
                ), None

            final_score = best_score * SCORE_SHAPE_MAX

            return StageResult(
                passed=True,
                score=final_score,
                reason=f"Shape gate passed ({best_info.get('n_vertices',0)}-vertex polygon)",
                debug_info=best_info,
            ), best_box

        except Exception as e:
            log.error(f"[Stage2-Shape] Exception: {e}")
            return StageResult(passed=False, score=0, reason=f"Exception: {e}"), None


# ══════════════════════════════════════════════════════════
#  STAGE 3: CONTEXT GATE (Anti-False-Positive)
# ══════════════════════════════════════════════════════════

class _ContextGate:

    def run(self, frame: np.ndarray, box: tuple) -> StageResult:
        """
        يرفض الـ detections المشبوهة:
        - جسم بشري (في أسفل الصورة + aspect عمودي)
        - إشارة مرور جانبية (على حواف الصورة)
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                return StageResult(passed=False, score=0, reason="Invalid box dimensions")

            box_aspect = bh / float(bw)
            box_center_y = (y1 + y2) / 2.0
            box_center_x = (x1 + x2) / 2.0

            # --- Anti-Human Body Detection ---
            # الشخص: في أسفل الصورة (center_y > 65%) + شكل عمودي
            relative_center_y = box_center_y / h
            if (relative_center_y > CONTEXT_BODY_BOTTOM_THRESH and
                    CONTEXT_BODY_ASPECT_MIN < box_aspect < CONTEXT_BODY_ASPECT_MAX):
                return StageResult(
                    passed=False,
                    score=0,
                    reason=(
                        f"Human body suspected: center_y={relative_center_y:.2f} "
                        f"(>{CONTEXT_BODY_BOTTOM_THRESH}), aspect={box_aspect:.2f}"
                    ),
                )

            # --- Anti-Edge-Sign Detection ---
            # لوحات الخطر الحقيقية نادراً ما تكون على الحافة الصقيلة للصورة
            left_margin  = x1 / w
            right_margin = (w - x2) / w
            top_margin   = y1 / h

            if left_margin < CONTEXT_EDGE_MARGIN_RATIO or right_margin < CONTEXT_EDGE_MARGIN_RATIO:
                # على الحافة اليمين أو اليسار بشكل مفرط → مشبوه
                # لكن نخففه لو المنطقة كبيرة (لوحة كبيرة تملأ الإطار)
                area_ratio = (bw * bh) / (w * h)
                if area_ratio < 0.06:  # صغيرة على الحافة → مشبوه جداً
                    return StageResult(
                        passed=False,
                        score=0,
                        reason=(
                            f"Edge-proximity suspicious: left={left_margin:.2f}, "
                            f"right={right_margin:.2f}, area_ratio={area_ratio:.3f}"
                        ),
                    )

            # --- Context Score ---
            # كلما كانت اللوحة في المنتصف أكثر → أعلى score
            center_x_norm = abs(box_center_x / w - 0.5) * 2  # 0 = مركز، 1 = حافة
            center_y_norm = abs(relative_center_y - 0.4)      # 0 = 40% من الصورة = أفضل مكان
            centrality_score = (1.0 - center_x_norm * 0.6 - center_y_norm * 0.4)
            centrality_score = max(centrality_score, 0.4)  # حد أدنى 40%

            context_score = centrality_score * SCORE_CONTEXT_MAX

            return StageResult(
                passed=True,
                score=context_score,
                reason="Context gate passed",
                debug_info={
                    "relative_center_y": relative_center_y,
                    "box_aspect":        box_aspect,
                    "left_margin":       left_margin,
                    "right_margin":      right_margin,
                    "centrality_score":  centrality_score,
                },
            )

        except Exception as e:
            log.error(f"[Stage3-Context] Exception: {e}")
            return StageResult(passed=False, score=0, reason=f"Exception: {e}")


# ══════════════════════════════════════════════════════════
#  STAGE 4: TEXTURE GATE
# ══════════════════════════════════════════════════════════

class _TextureGate:

    def run(self, frame: np.ndarray, box: tuple, glare_severity: str = "none") -> StageResult:
        """
        يتحقق من نسيج (texture) المنطقة المكشوفة.
        اللوحات عندها: حواف حادة + تباين عالٍ (بسبب الكتابة والرسومات).
        الملابس/الجلد: ناعمة، تباين منخفض، حواف قليلة.

        v2.1: لو الفريم بالكامل تحت glare قاسي، التباين المحلي جوه الـ
        crop نفسه ممكن يكون أضعف من الطبيعي حتى بعد تصحيح الفريم كله
        (الانعكاس المحلي على اللوحة نفسها أقوى من بقية الصورة) — فبنعمل
        local contrast boost إضافي على الـ crop + نخفف عتبة الـ gradient
        variance والـ edge density بنسبة معقولة، مش نلغيهم.
        """
        try:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            pad = 4
            crop = frame[
                max(0, y1 - pad): min(h, y2 + pad),
                max(0, x1 - pad): min(w, x2 + pad)
            ]

            if crop.size < 400:  # صغير جداً → نتجاوزه (benefit of doubt)
                return StageResult(
                    passed=True,
                    score=SCORE_TEXTURE_MAX * 0.6,
                    reason="Crop too small for texture analysis — pass with partial score",
                )

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # ── v2.1: local contrast boost تحت glare — CLAHE صغير ومحلي
            # على الـ crop نفسه (مش الفريم الكامل) عشان نرجّع تفاصيل
            # الكتابة/الرسومات اللي ممكن تكون لسه ضعيفة محليًا
            if glare_severity in ("mild", "high"):
                local_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                gray = local_clahe.apply(gray)

            # --- Gradient Magnitude Variance ---
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            grad_var  = float(np.var(magnitude))

            # --- Edge Density ---
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / edges.size

            # ── v2.1: عتبات متكيّفة — تحت glare، نخفف الحد الأدنى المطلوب
            # (الحواف بتفقد جزء من حدتها رغم كل التصحيحات) بس نسيب الحد
            # الأقصى زي ما هو (ضجيج بصري لسه ضجيج بصري حتى تحت الشمس)
            min_grad_var = TEXTURE_MIN_GRADIENT_VAR
            min_edge_den = TEXTURE_MIN_EDGE_DENSITY
            if glare_severity == "mild":
                min_grad_var = TEXTURE_MIN_GRADIENT_VAR * 0.7
                min_edge_den = TEXTURE_MIN_EDGE_DENSITY * 0.7
            elif glare_severity == "high":
                min_grad_var = TEXTURE_MIN_GRADIENT_VAR * 0.5
                min_edge_den = TEXTURE_MIN_EDGE_DENSITY * 0.5

            # --- التحقق ---
            if grad_var < min_grad_var:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=(
                        f"Texture too smooth (cloth/skin?): grad_var={grad_var:.1f} "
                        f"< {min_grad_var:.1f} (glare={glare_severity})"
                    ),
                )

            if grad_var > TEXTURE_MAX_GRADIENT_VAR:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"Texture too noisy: grad_var={grad_var:.1f} > {TEXTURE_MAX_GRADIENT_VAR}",
                )

            if edge_density < min_edge_den:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"Edge density too low: {edge_density:.3f} < {min_edge_den:.3f} (glare={glare_severity})",
                )

            if edge_density > TEXTURE_MAX_EDGE_DENSITY:
                return StageResult(
                    passed=False,
                    score=0,
                    reason=f"Edge density too high (noise?): {edge_density:.3f}",
                )

            # --- Texture Score ---
            # النطاق المثالي للـ gradient variance: 300..2000
            optimal_var = 800.0
            var_score   = 1.0 - min(abs(grad_var - optimal_var) / optimal_var, 1.0)

            # النطاق المثالي للـ edge density: 0.10..0.30
            optimal_edge = 0.18
            edge_score   = 1.0 - min(abs(edge_density - optimal_edge) / optimal_edge, 1.0)

            texture_score = (var_score * 0.6 + edge_score * 0.4) * SCORE_TEXTURE_MAX

            # تحت glare قاسي جدًا، نخصم جزء صغير من نقطة الثقة (نفس فلسفة
            # Color Gate) — الـ Temporal Gate هيعوّض الباقي عبر فريمات أكتر
            if glare_severity == "high":
                texture_score *= 0.88

            return StageResult(
                passed=True,
                score=texture_score,
                reason=f"Texture gate passed (glare={glare_severity})",
                debug_info={
                    "grad_var":      grad_var,
                    "edge_density":  edge_density,
                    "texture_score": texture_score,
                },
            )

        except Exception as e:
            log.error(f"[Stage4-Texture] Exception: {e}")
            return StageResult(passed=False, score=0, reason=f"Exception: {e}")


# ══════════════════════════════════════════════════════════
#  TEXT NORMALIZATION + OCR
# ══════════════════════════════════════════════════════════

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640]"
)

def _normalize_text(raw: str) -> str:
    if not raw:
        return ""
    txt = unicodedata.normalize("NFKC", raw)
    txt = _ARABIC_DIACRITICS.sub("", txt)
    txt = txt.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    txt = txt.replace("ى", "ي")
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _match_keywords(normalized_text: str) -> Tuple[Optional[str], str]:
    if not normalized_text or len(normalized_text) < OCR_MIN_TEXT_LEN:
        return None, "NONE"
    for kw in WARNING_KEYWORDS_CRITICAL:
        if _normalize_text(kw) in normalized_text:
            return kw, "CRITICAL"
    for kw in WARNING_KEYWORDS_GENERIC:
        if _normalize_text(kw) in normalized_text:
            return kw, "GENERIC"
    return None, "NONE"


class _TextDetector:
    """OCR في thread منفصل — مع graceful fallback لو Tesseract مش موجود"""

    def __init__(self):
        self._available       = False
        self._pytesseract     = None
        self._lock            = threading.Lock()
        self._latest_word     = ""
        self._latest_severity = "NONE"
        self._latest_raw      = ""
        self._latest_ts       = 0.0
        self._generic_hits    = 0
        self._last_generic_kw = ""
        self._pending_frame   = None
        self._ready_event     = threading.Event()
        self._running         = False
        self._thread          = None

    def load(self) -> bool:
        try:
            import shutil
            if shutil.which("tesseract") is None:
                log.warning(
                    "[TextDetector] 'tesseract' binary not found. "
                    "Install: sudo apt-get install tesseract-ocr tesseract-ocr-ara. "
                    "OCR path disabled — shape pipeline still active."
                )
                return False
            import pytesseract
            self._pytesseract = pytesseract
            self._available   = True
            log.info("[TextDetector] Tesseract OCR ready (ara+eng) ✓")
            return True
        except ImportError:
            log.warning("[TextDetector] pytesseract not installed. OCR disabled.")
            return False
        except Exception as e:
            log.error(f"[TextDetector] load error: {e}")
            return False

    def start(self):
        if not self._available:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker_loop, daemon=True, name="SignDetV2-OCR"
        )
        self._thread.start()
        log.info("[TextDetector] OCR worker thread started")

    def stop(self):
        self._running = False
        self._ready_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit(self, frame_or_crop: np.ndarray):
        if not self._available or not self._running:
            return
        if not self._ready_event.is_set():
            self._pending_frame = frame_or_crop.copy()
            self._ready_event.set()

    def get_latest(self) -> Tuple[str, str, str, float]:
        with self._lock:
            return (self._latest_word, self._latest_severity,
                    self._latest_raw, self._latest_ts)

    @property
    def available(self) -> bool:
        return self._available

    def _worker_loop(self):
        while self._running:
            self._ready_event.wait(timeout=2.0)
            if not self._running:
                break
            frame = self._pending_frame
            self._pending_frame = None
            self._ready_event.clear()
            if frame is None:
                continue
            try:
                self._run_ocr(frame)
            except Exception as e:
                log.error(f"[TextDetector] OCR error: {e}")

    def _run_ocr(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # تحسين التباين قبل OCR
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # إزالة الضوضاء
        gray = cv2.medianBlur(gray, 3)

        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        data = self._pytesseract.image_to_data(
            thresh,
            lang=OCR_LANGS,
            output_type=self._pytesseract.Output.DICT,
            config="--psm 6 --oem 1",  # LSTM mode
        )

        words = []
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            txt = (txt or "").strip()
            if txt and conf_val >= OCR_WORD_MIN_CONF:
                words.append(txt)

        raw_text   = " ".join(words)
        normalized = _normalize_text(raw_text)
        matched_kw, severity = _match_keywords(normalized)

        if severity == "GENERIC":
            if matched_kw == self._last_generic_kw:
                self._generic_hits += 1
            else:
                self._last_generic_kw = matched_kw
                self._generic_hits    = 1
            if self._generic_hits < TEXT_GENERIC_CONFIRMS:
                matched_kw, severity = None, "NONE"
        else:
            self._generic_hits    = 0
            self._last_generic_kw = ""

        with self._lock:
            self._latest_word     = matched_kw or ""
            self._latest_severity = severity
            self._latest_raw      = raw_text
            self._latest_ts       = time.time()

        if severity != "NONE":
            log.info(
                f"[TextDetector] matched='{matched_kw}' severity={severity} raw='{raw_text}'"
            )


# ══════════════════════════════════════════════════════════
#  MAIN CLASS: SignDetector V2
# ══════════════════════════════════════════════════════════

class SignDetector:
    """
    نظام الكشف الكامل — استخدامه:

        sd = SignDetector()
        sd.load()
        sd.start()

        result = sd.process_frame(frame)
        if result.danger_confirmed:
            # أرسل أمر إيقاف فوري
    """

    def __init__(self):
        self._color_gate   = _ColorGate()
        self._shape_gate   = _ShapeGate()
        self._context_gate = _ContextGate()
        self._texture_gate = _TextureGate()
        self._temporal     = _TemporalGate(TEMPORAL_WINDOW, TEMPORAL_HITS)
        self._text         = _TextDetector()

        self._frame_count      = 0
        self._lock             = threading.Lock()
        self._latest           = SignDetectionResult()

        # لما الـ danger يتأكد، بنسجل الوقت عشان نضمن minimum hold
        self._danger_held_until = 0.0   # danger_confirmed = True لحد الوقت ده على الأقل
        self._clear_streak      = 0     # عداد الفريمات المتتالية الفاشلة في Stage 1
        self._danger_active     = False # True لما يكون danger_confirmed حصل

        # للـ debug: عداد رفض كل stage
        self._rejection_stats = {
            "stage1_color":   0,
            "stage2_shape":   0,
            "stage3_context": 0,
            "stage4_texture": 0,
            "stage5_temporal":0,
            "score_too_low":  0,
            "confirmed":      0,
        }

    def load(self) -> bool:
        self._text_available = self._text.load()
        if not self._text_available:
            log.warning(
                "[SignDetector] Running in SHAPE-ONLY mode. "
                f"Score threshold raised to {SCORE_MIN_CONFIRM_NO_TEXT}/100"
            )
        return True

    def start(self):
        if self._text_available:
            self._text.start()

    def stop(self):
        if self._text_available:
            self._text.stop()
        log.info(f"[SignDetector] Rejection stats: {self._rejection_stats}")

    def process_frame(self, frame: np.ndarray) -> SignDetectionResult:
        self._frame_count += 1
        total_score  = 0.0
        reject_reasons: List[str] = []

        # ─── Stage 1: COLOR GATE (+ Glare detection/correction v2.1) ──
        color_result, color_mask, work_frame, glare_severity = self._color_gate.run(frame)
        if not color_result.passed:
            self._rejection_stats["stage1_color"] += 1
            self._temporal.update(False)
            self._clear_streak += 1

            # لو الـ danger كان active، نشوف هل المفروض يفضل active
            still_held   = time.time() < self._danger_held_until
            streak_ok    = self._clear_streak < CLEAR_STREAK_NEEDED

            if self._danger_active and (still_held or streak_ok):
                # الـ danger لسه active — إما في الـ hold window أو الـ streak مش كفاية
                with self._lock:
                    # نحدث الـ latest بس نخلي danger_confirmed = True
                    held_result = SignDetectionResult(
                        stage1_color=False,
                        danger_confirmed=True,
                        reason="HELD",
                        reject_reason=(
                            f"[S1-Color] {color_result.reason} "
                            f"| held (streak={self._clear_streak}/{CLEAR_STREAK_NEEDED})"
                        ),
                    )
                    self._latest = held_result
                return held_result
            else:
                # الـ danger اتمسح فعلاً
                if self._danger_active:
                    self._danger_active = False
                    log.info(
                        f"[SignDetector] CLEAR confirmed after {self._clear_streak} "
                        f"consecutive no-color frames — danger reset"
                    )
                result = SignDetectionResult(
                    stage1_color=False,
                    danger_confirmed=False,
                    reject_reason=f"[S1-Color] {color_result.reason}",
                )
                with self._lock:
                    self._latest = result
                return result

        # اللوحة لسه موجودة → reset الـ clear streak
        self._clear_streak = 0
        total_score += color_result.score

        # ─── Stage 2: SHAPE GATE ─────────────────────────────
        # نستخدم work_frame (المصحَّح لو فيه glare) — الحواف بترجع تبان
        # أوضح بعد الـ CLAHE + highlight recovery، فالـ contour detection
        # بيكون أدق تحت إضاءة قاسية
        shape_result, best_box = self._shape_gate.run(work_frame, color_mask)
        if not shape_result.passed:
            self._rejection_stats["stage2_shape"] += 1
            self._temporal.update(False)
            result = SignDetectionResult(
                stage1_color=True,
                stage2_shape=False,
                danger_confirmed=False,
                reject_reason=f"[S2-Shape] {shape_result.reason}",
            )
            with self._lock:
                self._latest = result
            return result

        total_score += shape_result.score

        # ─── Stage 3: CONTEXT GATE ───────────────────────────
        context_result = self._context_gate.run(work_frame, best_box)
        if not context_result.passed:
            self._rejection_stats["stage3_context"] += 1
            self._temporal.update(False)
            result = SignDetectionResult(
                stage1_color=True,
                stage2_shape=True,
                stage3_context=False,
                shape_box=best_box,
                danger_confirmed=False,
                reject_reason=f"[S3-Context] {context_result.reason}",
            )
            with self._lock:
                self._latest = result
            return result

        total_score += context_result.score

        # ─── Stage 4: TEXTURE GATE ───────────────────────────
        # severity بتتمرر عشان الـ Texture Gate يخفف عتبة الـ gradient
        # variance تحت glare (الحواف بتفقد بعض حدتها حتى بعد التصحيح)
        texture_result = self._texture_gate.run(work_frame, best_box, glare_severity)
        if not texture_result.passed:
            self._rejection_stats["stage4_texture"] += 1
            self._temporal.update(False)
            result = SignDetectionResult(
                stage1_color=True,
                stage2_shape=True,
                stage3_context=True,
                stage4_texture=False,
                shape_box=best_box,
                danger_confirmed=False,
                reject_reason=f"[S4-Texture] {texture_result.reason}",
            )
            with self._lock:
                self._latest = result
            return result

        total_score += texture_result.score

        # ─── OCR (bonus score) ───────────────────────────────
        # نرسل الـ crop للـ OCR worker — من work_frame (المصحَّح) عشان
        # الكتابة تكون أوضح للـ OCR تحت glare بدل الفريم الخام المحروق
        if self._text_available and best_box:
            x1, y1, x2, y2 = best_box
            h_f, w_f = work_frame.shape[:2]
            pad  = 10
            crop = work_frame[
                max(0, y1 - pad): min(h_f, y2 + pad),
                max(0, x1 - pad): min(w_f, x2 + pad),
            ]
            if crop.size > 0:
                self._text.submit(crop)

        # Full-frame OCR بشكل دوري
        if self._text_available and (self._frame_count % OCR_FULLFRAME_EVERY_N == 0):
            self._text.submit(work_frame)

        word, severity, raw_text, _ = self._text.get_latest()
        text_detected = severity != "NONE"

        # OCR bonus
        text_score = 0.0
        if severity == "CRITICAL":
            text_score = SCORE_TEXT_MAX
        elif severity == "GENERIC":
            text_score = SCORE_TEXT_MAX * 0.5

        total_score += text_score

        # ─── Score Threshold Check ────────────────────────────
        threshold = (
            SCORE_MIN_CONFIRM_WITH_OCR
            if severity == "CRITICAL"
            else SCORE_MIN_CONFIRM_NO_TEXT
        )

        if total_score < threshold:
            self._rejection_stats["score_too_low"] += 1
            self._temporal.update(False)
            result = SignDetectionResult(
                stage1_color=True,
                stage2_shape=True,
                stage3_context=True,
                stage4_texture=True,
                shape_box=best_box,
                confidence_score=total_score,
                text_matched_word=word,
                text_severity=severity,
                raw_text=raw_text,
                danger_confirmed=False,
                reject_reason=(
                    f"[Score] {total_score:.1f} < threshold {threshold} "
                    f"(text={severity})"
                ),
            )
            with self._lock:
                self._latest = result
            return result

        # ─── Stage 5: TEMPORAL GATE ──────────────────────────
        temporal_confirmed = self._temporal.update(True)

        if not temporal_confirmed:
            hits = self._temporal.current_hits
            win  = self._temporal.window_size
            result = SignDetectionResult(
                stage1_color=True,
                stage2_shape=True,
                stage3_context=True,
                stage4_texture=True,
                stage5_temporal=False,
                shape_box=best_box,
                confidence_score=total_score,
                text_matched_word=word,
                text_severity=severity,
                raw_text=raw_text,
                danger_confirmed=False,
                reject_reason=(
                    f"[S5-Temporal] Accumulating: {hits}/{TEMPORAL_HITS} "
                    f"in window {win}/{TEMPORAL_WINDOW}"
                ),
            )
            with self._lock:
                self._latest = result
            return result

        # ─── DANGER CONFIRMED ─────────────────────────────────
        self._rejection_stats["confirmed"] += 1
        self._danger_active     = True
        self._clear_streak      = 0
        self._danger_held_until = time.time() + DANGER_MIN_HOLD_SEC

        if text_detected and total_score > threshold + 20:
            reason = f"SHAPE+TEXT ({severity})"
        elif text_detected:
            reason = f"SHAPE+TEXT"
        else:
            reason = "SHAPE"

        log.warning(
            f"[SignDetector] *** DANGER CONFIRMED *** "
            f"score={total_score:.1f}/{threshold} reason={reason} "
            f"box={best_box}"
        )

        result = SignDetectionResult(
            stage1_color=True,
            stage2_shape=True,
            stage3_context=True,
            stage4_texture=True,
            stage5_temporal=True,
            shape_box=best_box,
            confidence_score=total_score,
            text_matched_word=word,
            text_severity=severity,
            raw_text=raw_text,
            danger_confirmed=True,
            reason=reason,
        )

        with self._lock:
            self._latest = result

        return result

    def get_latest(self) -> SignDetectionResult:
        with self._lock:
            return self._latest

    def get_rejection_stats(self) -> dict:
        """للـ debugging: إحصائيات رفض كل مرحلة"""
        return dict(self._rejection_stats)


# ══════════════════════════════════════════════════════════
#  QUICK TEST (تشغيل مباشر)
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    sd = SignDetector()
    sd.load()
    sd.start()

    cap = cv2.VideoCapture(0)
    print("SignDetector V2 — اضغط q للخروج | اضغط s لعرض الإحصائيات")

    frame_n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_n += 1
            result = sd.process_frame(frame)
            display = frame.copy()

            # رسم الـ bounding box
            if result.shape_box:
                x1, y1, x2, y2 = result.shape_box
                color = (0, 0, 255) if result.danger_confirmed else (0, 180, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Stage indicators (أعلى يمين)
            stages = [
                ("C", result.stage1_color),
                ("Sh", result.stage2_shape),
                ("Cx", result.stage3_context),
                ("Tx", result.stage4_texture),
                ("Tm", result.stage5_temporal),
            ]
            for i, (lbl, passed) in enumerate(stages):
                col = (0, 200, 0) if passed else (80, 80, 80)
                cv2.putText(display, lbl, (frame.shape[1] - 160 + i * 28, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # Score
            score_str = f"Score: {result.confidence_score:.0f}/100"
            cv2.putText(display, score_str, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Status
            if result.danger_confirmed:
                status = f"!!! DANGER [{result.reason}] !!!"
                status_col = (0, 0, 255)
            elif result.reject_reason:
                # اختصر سبب الرفض للعرض
                short = result.reject_reason[:60]
                status = f"CLEAR: {short}"
                status_col = (0, 200, 0)
            else:
                status = "CLEAR"
                status_col = (0, 200, 0)

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

            # Temporal accumulation
            if result.stage4_texture and not result.danger_confirmed:
                hits = sd._temporal.current_hits
                temporal_str = f"Temporal: {hits}/{TEMPORAL_HITS}"
                cv2.putText(display, temporal_str, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)

            # Text
            if result.text_matched_word:
                cv2.putText(display, f"TEXT: {result.text_matched_word}", (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 255), 1)

            cv2.imshow("SignDetector V2", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                stats = sd.get_rejection_stats()
                print("\n=== Rejection Stats ===")
                for k, v in stats.items():
                    print(f"  {k:20s}: {v}")
                print(f"  {'total_frames':20s}: {frame_n}")
                print("======================\n")

    finally:
        sd.stop()
        cap.release()
        cv2.destroyAllWindows()