from __future__ import annotations

import math
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.schemas.chat import SourceItem
from app.schemas.transformer import (
    TransformerChatRequest,
    TransformerClassifyItem,
    TransformerClassifyRequest,
    TransformerClusterGroup,
    TransformerClusterRequest,
    TransformerEvalRequest,
    TransformerQuickTestItem,
    TransformerQuickTestRequest,
    TransformerRagAnalyzeRequest,
    TransformerTopicTemplate,
)
from app.services.embedding_service import (
    embed_texts,
    normalize_embedding_model_name,
)
from app.services.model_router_service import generate_answer_by_provider
from app.services.retrieval_service import hybrid_retrieve
from app.services.local_transformer_service import local_transformer_runtime


TOPIC_TEMPLATES: list[TransformerTopicTemplate] = [
    TransformerTopicTemplate(
        code="02",
        title="词元（Token）与嵌入（Embedding）",
        prompt="解释词元切分、向量嵌入及其对检索质量和生成效果的影响。",
        keywords=["词元", "嵌入", "向量", "检索"],
    ),
    TransformerTopicTemplate(
        code="03",
        title="LLM 的内部机制",
        prompt="说明自注意力、前馈网络、位置编码和解码流程。",
        keywords=["注意力", "位置编码", "前馈", "解码"],
    ),
    TransformerTopicTemplate(
        code="04",
        title="文本分类",
        prompt="给出文本分类任务的标签设计、训练流程和评估方法。",
        keywords=["标签", "训练", "准确率", "F1"],
    ),
    TransformerTopicTemplate(
        code="05",
        title="文本聚类",
        prompt="说明文本聚类流程、簇解释和典型应用场景。",
        keywords=["聚类", "簇", "主题", "相似度"],
    ),
    TransformerTopicTemplate(
        code="06",
        title="提示工程",
        prompt="给出高质量提示模板设计方法和防幻觉策略。",
        keywords=["提示", "模板", "约束", "幻觉"],
    ),
    TransformerTopicTemplate(
        code="07",
        title="高级文本生成技术",
        prompt="说明结构化生成、长文本生成与自检机制。",
        keywords=["生成", "结构化", "长文本", "自检"],
    ),
    TransformerTopicTemplate(
        code="08",
        title="语义搜索与RAG",
        prompt="说明RAG召回、重排、引用追踪和失败回退策略。",
        keywords=["RAG", "召回", "重排", "引用"],
    ),
    TransformerTopicTemplate(
        code="09",
        title="构建文本嵌入模型",
        prompt="说明嵌入模型构建流程、数据准备和对比学习。",
        keywords=["嵌入模型", "对比学习", "语料", "召回"],
    ),
    TransformerTopicTemplate(
        code="10",
        title="为分类任务微调表示模型",
        prompt="说明表示模型微调方案、损失函数与线上部署。",
        keywords=["微调", "表示模型", "损失函数", "部署"],
    ),
    TransformerTopicTemplate(
        code="11",
        title="微调生成模型",
        prompt="说明SFT/LoRA训练流程、数据规范与安全策略。",
        keywords=["SFT", "LoRA", "指令数据", "安全"],
    ),
    TransformerTopicTemplate(
        code="12",
        title="Transformer 与大模型应用开发提升",
        prompt="给出面向业务落地的架构、评估与迭代方案。",
        keywords=["架构", "评估", "迭代", "落地"],
    ),
]


def _latest_user_question(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def transformer_chat(
    db: Session, payload: TransformerChatRequest
) -> tuple[str, str, list[SourceItem], dict[str, int | float | str | bool]]:
    total_start = perf_counter()
    question = _latest_user_question([item.model_dump() for item in payload.messages])
    if not question:
        question = "请根据知识库给出专题分析。"

    kb_ids = payload.kb_ids or [
        item.id
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.status == "active")
        ).all()
    ]

    sources: list[SourceItem] = []
    contexts: list[str] = []
    retrieval_ms = 0
    retrieval_fallback = False
    if payload.rag_enabled:
        retrieval_start = perf_counter()
        try:
            retrieved = hybrid_retrieve(
                db=db,
                query=question,
                kb_ids=kb_ids,
                document_ids=payload.document_ids,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                fusion_mode=payload.fusion_mode,
                alpha=payload.alpha,
            )
        except Exception:
            retrieved = []
            retrieval_fallback = True
        retrieval_ms = int((perf_counter() - retrieval_start) * 1000)
        contexts = [item["content"] for item in retrieved]
        sources = [
            SourceItem(
                source_location=item["source_location"],
                content=item["content"],
                score=round(float(item["score"]), 4),
            )
            for item in retrieved
        ]

    answer, used_model, generation_metrics = generate_answer_by_provider(
        provider=payload.provider,
        question=question,
        contexts=contexts,
        model=payload.model,
        temperature=payload.temperature,
        max_new_tokens=payload.max_new_tokens,
    )
    diagnostics: dict[str, int | float | str | bool] = {
        "rag_enabled": payload.rag_enabled,
        "retrieved_chunks": len(contexts),
        "retrieval_ms": retrieval_ms,
        "retrieval_fallback": retrieval_fallback,
        "total_ms": int((perf_counter() - total_start) * 1000),
    }
    diagnostics.update(generation_metrics)
    return answer, used_model, sources, diagnostics


def transformer_classify(
    payload: TransformerClassifyRequest,
) -> tuple[str, list[TransformerClassifyItem]]:
    model_name = normalize_embedding_model_name(payload.model)
    text_vectors = embed_texts(payload.texts, model_name=model_name)
    label_vectors = embed_texts(payload.labels, model_name=model_name)

    items: list[TransformerClassifyItem] = []
    for text, text_vector in zip(payload.texts, text_vectors):
        ranking_pairs: list[tuple[str, float]] = []
        for label, label_vector in zip(payload.labels, label_vectors):
            ranking_pairs.append((label, _cosine_similarity(text_vector, label_vector)))
        ranking_pairs.sort(key=lambda item: item[1], reverse=True)
        top_label, top_score = ranking_pairs[0]
        items.append(
            TransformerClassifyItem(
                text=text,
                label=top_label,
                score=round(float(top_score), 6),
                ranking=[
                    {label: round(float(score), 6)} for label, score in ranking_pairs
                ],
            )
        )
    return model_name, items


def _avg_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    sums = [0.0] * dims
    for vec in vectors:
        for idx in range(dims):
            sums[idx] += vec[idx]
    return [value / len(vectors) for value in sums]


def _nearest_center(vector: list[float], centers: list[list[float]]) -> int:
    best_idx = 0
    best_score = -2.0
    for idx, center in enumerate(centers):
        score = _cosine_similarity(vector, center)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def transformer_cluster(
    payload: TransformerClusterRequest,
) -> tuple[str, list[int], list[TransformerClusterGroup]]:
    model_name = normalize_embedding_model_name(payload.model)
    vectors = embed_texts(payload.texts, model_name=model_name)
    k = min(payload.k, len(vectors))
    centers = [vectors[idx][:] for idx in range(k)]
    assignments = [0] * len(vectors)

    for _ in range(payload.max_iter):
        changed = False
        for idx, vec in enumerate(vectors):
            cluster_id = _nearest_center(vec, centers)
            if assignments[idx] != cluster_id:
                assignments[idx] = cluster_id
                changed = True
        if not changed:
            break

        for cluster_id in range(k):
            member_vectors = [
                vec
                for member_idx, vec in enumerate(vectors)
                if assignments[member_idx] == cluster_id
            ]
            if member_vectors:
                centers[cluster_id] = _avg_vector(member_vectors)

    groups: list[TransformerClusterGroup] = []
    for cluster_id in range(k):
        member_indices = [
            idx for idx, value in enumerate(assignments) if value == cluster_id
        ]
        sample_texts = [payload.texts[idx] for idx in member_indices[:5]]
        groups.append(
            TransformerClusterGroup(
                cluster_id=cluster_id,
                size=len(member_indices),
                sample_texts=sample_texts,
            )
        )
    return model_name, assignments, groups


def transformer_rag_analyze(
    db: Session, payload: TransformerRagAnalyzeRequest
) -> tuple[str, str, list[SourceItem], dict[str, int | float | str | bool]]:
    total_start = perf_counter()
    retrieval_start = perf_counter()
    kb_ids = payload.kb_ids or [
        item.id
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.status == "active")
        ).all()
    ]
    retrieved = hybrid_retrieve(
        db=db,
        query=payload.topic,
        kb_ids=kb_ids,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        fusion_mode=payload.fusion_mode,
        alpha=payload.alpha,
    )
    retrieval_ms = int((perf_counter() - retrieval_start) * 1000)
    contexts = [item["content"] for item in retrieved]
    sources = [
        SourceItem(
            source_location=item["source_location"],
            content=item["content"],
            score=round(float(item["score"]), 4),
        )
        for item in retrieved
    ]
    aspect_text = (
        "、".join(payload.aspects) if payload.aspects else "知识点梳理、应用建议"
    )
    prompt = (
        f"请围绕主题《{payload.topic}》做分析，重点覆盖：{aspect_text}。"
        "输出格式：1)核心概念 2)实践建议 3)常见误区。"
    )
    answer, used_model, generation_metrics = generate_answer_by_provider(
        provider=payload.provider,
        question=prompt,
        contexts=contexts,
        model=payload.model,
    )
    diagnostics: dict[str, int | float | str | bool] = {
        "retrieved_chunks": len(contexts),
        "retrieval_ms": retrieval_ms,
        "total_ms": int((perf_counter() - total_start) * 1000),
    }
    diagnostics.update(generation_metrics)
    return answer, used_model, sources, diagnostics


def transformer_eval(
    payload: TransformerEvalRequest,
) -> tuple[str, int, int, float, float]:
    start = perf_counter()
    classify_model, items = transformer_classify(
        TransformerClassifyRequest(
            texts=[item.text for item in payload.samples],
            labels=payload.labels,
            model=payload.model,
        )
    )
    total = len(payload.samples)
    correct = 0
    for item, sample in zip(items, payload.samples):
        if item.label == sample.expected_label:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    elapsed_ms = (perf_counter() - start) * 1000
    return classify_model, total, correct, accuracy, elapsed_ms


def transformer_runtime() -> dict[str, str | bool | int]:
    runtime = local_transformer_runtime()
    runtime["embedding_model"] = settings.default_embedding_model
    return runtime


def list_topic_templates() -> list[TransformerTopicTemplate]:
    return TOPIC_TEMPLATES


def _calc_keyword_score(answer: str, keywords: list[str]) -> tuple[int, int, float]:
    if not keywords:
        return 0, 0, 0.0
    hits = 0
    lowered = answer.lower()
    for keyword in keywords:
        if keyword and keyword.lower() in lowered:
            hits += 1
    score = (hits / len(keywords)) * 100.0
    return hits, len(keywords), score


def quick_test_topics(
    db: Session, payload: TransformerQuickTestRequest
) -> tuple[str, int, int, float, list[TransformerQuickTestItem]]:
    selected = TOPIC_TEMPLATES
    if payload.topic_codes:
        selected_codes = {code.strip() for code in payload.topic_codes if code.strip()}
        selected = [item for item in TOPIC_TEMPLATES if item.code in selected_codes]

    if not selected:
        return "", 0, 0, 0.0, []

    selected = selected[: payload.max_topics]

    items: list[TransformerQuickTestItem] = []
    total_score = 0.0
    pass_count = 0
    used_model = payload.model or settings.local_transformer_model

    for template in selected:
        start = perf_counter()
        topic_query = f"{template.code}. {template.title}。{template.prompt}"
        try:
            retrieved = hybrid_retrieve(
                db=db,
                query=topic_query,
                kb_ids=payload.kb_ids
                or [
                    item.id
                    for item in db.scalars(
                        select(KnowledgeBase).where(KnowledgeBase.status == "active")
                    ).all()
                ],
                document_ids=payload.document_ids,
                top_k=max(2, min(6, payload.top_k)),
                score_threshold=payload.score_threshold,
                fusion_mode=payload.fusion_mode,
                alpha=payload.alpha,
            )
            contexts = [item.get("content", "") for item in retrieved[:3]]
        except Exception:
            contexts = []

        mode = "generation" if payload.run_generation else "retrieval"
        if payload.run_generation:
            analysis, model, _ = generate_answer_by_provider(
                provider=payload.provider,
                question=topic_query,
                contexts=contexts,
                model=payload.model,
                max_new_tokens=128,
                temperature=0.1,
            )
            used_model = model
        else:
            analysis = "\n".join(contexts).strip()[:400]
            if not analysis:
                mode = "template_baseline"
                analysis = template.prompt

        hits, total_keywords, score = _calc_keyword_score(analysis, template.keywords)
        penalty = 1.0
        if not contexts:
            penalty *= 0.6
        if not payload.run_generation:
            penalty *= 0.9
        score = score * penalty
        latency_ms = int((perf_counter() - start) * 1000)
        passed = score >= payload.pass_threshold
        if passed:
            pass_count += 1
        total_score += score
        items.append(
            TransformerQuickTestItem(
                code=template.code,
                title=template.title,
                prompt=template.prompt,
                score=round(score, 2),
                passed=passed,
                keyword_hits=hits,
                total_keywords=total_keywords,
                retrieved_chunks=len(contexts),
                mode=mode,
                latency_ms=latency_ms,
                answer_preview=analysis[:300],
            )
        )

    avg_score = total_score / len(items)
    return used_model, len(items), pass_count, avg_score, items


def build_quick_test_markdown_report(
    provider: str,
    model: str,
    total_topics: int,
    pass_count: int,
    average_score: float,
    items: list[TransformerQuickTestItem],
) -> str:
    lines: list[str] = []
    lines.append("# 西交 Transformer 专题评测报告")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- Provider: `{provider}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Topic Count: `{total_topics}`")
    lines.append(f"- Pass Count: `{pass_count}`")
    lines.append(f"- Average Score: `{round(float(average_score), 2)}`")
    lines.append("")
    lines.append("## 分项结果")
    lines.append("")
    lines.append(
        "| 编号 | 专题 | 得分 | 通过 | 关键词命中 | 检索块 | 模式 | 时延(ms) |"
    )
    lines.append("|---|---|---:|:---:|---:|---:|---|---:|")
    for item in items:
        lines.append(
            "| "
            f"{item.code} | {item.title} | {item.score} | "
            f"{'是' if item.passed else '否'} | {item.keyword_hits}/{item.total_keywords} | "
            f"{item.retrieved_chunks} | {item.mode} | {item.latency_ms} |"
        )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    if total_topics > 0:
        pass_rate = (pass_count / total_topics) * 100.0
    else:
        pass_rate = 0.0
    lines.append(f"- 通过率：`{round(pass_rate, 2)}%`")
    lines.append("- 建议优先关注低分专题，补充对应知识库文档并复测。")
    lines.append("- 若需真实性能评估，可开启 run_generation=true 进行生成模式测试。")
    lines.append("")
    return "\n".join(lines)
