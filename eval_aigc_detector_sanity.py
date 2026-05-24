from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.aigc_detector.onnx_detector import ModelSpec, OnnxAigcDetector


@dataclass(frozen=True)
class TestText:
    name: str
    text: str


def _contains_cjk(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7AF
        ):
            return True
    return False


def _mk_detector(models_dir: Path, cache_dir: Path, repo_id: str, onnx_path: str, max_length: int) -> OnnxAigcDetector:
    return OnnxAigcDetector(
        models_dir=models_dir,
        cache_dir=cache_dir,
        endpoint="https://hf-mirror.com",  # unused if files exist locally
        spec=ModelSpec(repo_id=repo_id, onnx_path=onnx_path, max_length=max_length),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check AIGC detector models with long-ish texts")
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    ap.add_argument("--max-len", type=int, default=256, help="Tokenizer max length (model input length)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    cache_dir = Path(args.cache_dir)

    # Current defaults in scripts/run_aigc_detector.ps1
    det_en = _mk_detector(
        models_dir,
        cache_dir,
        repo_id="onnx-community/answerdotai-ModernBERT-base-ai-detector-ONNX",
        onnx_path="onnx/model_int8.onnx",
        max_length=args.max_len,
    )

    # Alternate English ONNX detector you already vendored (useful for comparison)
    det_en_roberta = _mk_detector(
        models_dir,
        cache_dir,
        repo_id="onnx-community/chatgpt-detector-roberta-ONNX",
        onnx_path="onnx/model_int8.onnx",
        max_length=args.max_len,
    )

    # Current default zh ONNX detector in scripts/run_aigc_detector.ps1
    det_zh = _mk_detector(
        models_dir,
        cache_dir,
        repo_id="ZTL-UwU/chatgpt-detector-roberta-chinese-onnx",
        onnx_path="model.onnx",
        max_length=args.max_len,
    )

    tests: list[TestText] = [
        TestText(
            name="en_human_long",
            text=(
                "I wrote this after a long day of debugging and it reads like a real note to myself. "
                "The problem looked simple at first: a small change to a template caused a subtle crash, "
                "but only when a particular engine returned metadata in a different shape. "
                "I spent the morning reproducing it, then the afternoon narrowing it down to a single assumption. "
                "Once I stopped guessing and started checking types, everything became clearer. "
                "I also noticed how my own writing wanders a bit: I repeat phrases, I correct myself, "
                "and I sometimes leave a thought unfinished before returning to it. "
                "By the time I fixed it, I had a short list of follow-ups: add a guard, add a test, "
                "and document the behavior so I do not forget it next week. "
                "If you are reading this, you are probably me, and you are probably tired. "
                "Get some water, then run the check again and make sure the change is actually minimal. "
                "The goal is not to be clever, it is to be correct. "
                "Finally, I want to remind myself that good engineering is mostly careful verification: "
                "measure first, change second, and only then optimize."
            ),
        ),
        TestText(
            name="en_ai_long",
            text=(
                "As an AI language model, I can provide a structured overview of the situation. "
                "First, we will identify the key requirements and constraints. "
                "Second, we will propose a step-by-step implementation plan. "
                "Third, we will validate the results using reproducible checks. "
                "It is important to consider performance, correctness, and maintainability. "
                "In many cases, a robust solution should include clear error handling and logging. "
                "Additionally, it is recommended to minimize unrelated changes to facilitate code review. "
                "If the model appears to perform poorly, common causes include insufficient input length, "
                "domain mismatch, or incorrect interpretation of model outputs. "
                "To address these issues, we can adjust preprocessing, increase the amount of text analyzed, "
                "or evaluate alternative models. "
                "In conclusion, by following best practices and iterating based on evidence, "
                "we can improve both user experience and system reliability."
            ),
        ),
        TestText(
            name="zh_human_long",
            text=(
                "这段文字是我自己写的随手记录，读起来更像是一个真实的人在复盘一天的工作。"
                "一开始我以为问题出在模型上，因为分数看起来很不稳定，后来才发现其实是输入太短。"
                "搜索结果只有标题和摘要，往往也就一两句话，模型就算很强也很难给出可靠判断。"
                "我试着把几段更长的内容拼起来再测，分布才稍微正常一点。"
                "当然，这也带来另一个麻烦：如果我们真的去抓取网页全文，会变慢，而且会多一次网络请求。"
                "我比较倾向于先把现有链路做对：别把分数用反，别在模板里假设字段一定是字符串，"
                "再把偏好项做成可以调节的强度，让用户自己决定影响有多大。"
                "等这些都稳定了，再考虑要不要加更激进的做法。"
            ),
        ),
        TestText(
            name="zh_ai_long",
            text=(
                "作为一个大型语言模型，我将以条理清晰的方式对当前问题进行说明。"
                "首先，我们需要确认模型推理流程是否正确，包括分词、截断、以及输出概率的解释。"
                "其次，我们需要评估输入文本长度对检测效果的影响，并在必要时采用分段策略。"
                "第三，我们应当确保界面提示与排序逻辑一致，避免用户产生误解。"
                "此外，还应当尽量减少对上游项目的无关修改，以便顺利提交并通过代码审查。"
                "最后，通过对比测试与持续迭代，我们可以提升系统的稳定性与可用性。"
            ),
        ),
        TestText(name="en_short", text="This is a short sentence about the weather."),
        TestText(name="zh_short", text="这是一句很短的话。"),
    ]

    dets = [
        ("en_modernbert", det_en),
        ("en_roberta", det_en_roberta),
        ("zh_roberta", det_zh),
    ]

    print(f"max_len={args.max_len} threshold={args.threshold}")
    print("name\tchars\tcjk\tmodel\tscore_ai\tlabel")

    for test in tests:
        cjk = _contains_cjk(test.text)
        # Route by language: for CJK texts, evaluate the zh model; for non-CJK, evaluate both EN models
        for model_name, det in dets:
            if cjk and not model_name.startswith("zh_"):
                continue
            if (not cjk) and model_name.startswith("zh_"):
                continue

            s = det.score_ai([test.text])[0]
            label = det.label_from_score(s, threshold=args.threshold)
            print(f"{test.name}\t{len(test.text)}\t{int(cjk)}\t{model_name}\t{s:.4f}\t{label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
