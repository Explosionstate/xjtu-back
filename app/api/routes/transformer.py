from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.transformer import (
    TransformerChatRequest,
    TransformerChatResponse,
    TransformerClassifyRequest,
    TransformerClassifyResponse,
    TransformerClusterRequest,
    TransformerClusterResponse,
    TransformerEvalRequest,
    TransformerEvalResponse,
    TransformerQuickTestRequest,
    TransformerQuickTestResponse,
    TransformerRagAnalyzeRequest,
    TransformerRagAnalyzeResponse,
    TransformerTopicTemplateResponse,
    TransformerRuntimeResponse,
)
from app.services.transformer_task_service import (
    build_quick_test_markdown_report,
    list_topic_templates,
    quick_test_topics,
    transformer_chat,
    transformer_classify,
    transformer_cluster,
    transformer_eval,
    transformer_rag_analyze,
    transformer_runtime,
)

router = APIRouter(prefix="/transformer", tags=["transformer"])


@router.get("/runtime", response_model=TransformerRuntimeResponse)
def runtime(_: object = Depends(get_current_user)) -> TransformerRuntimeResponse:
    return TransformerRuntimeResponse.model_validate(transformer_runtime())


@router.get("/topics/templates", response_model=TransformerTopicTemplateResponse)
def topics_templates(
    _: object = Depends(get_current_user),
) -> TransformerTopicTemplateResponse:
    return TransformerTopicTemplateResponse(items=list_topic_templates())


@router.post("/chat/completions", response_model=TransformerChatResponse)
def chat(
    payload: TransformerChatRequest,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
) -> TransformerChatResponse:
    answer, model, sources, diagnostics = transformer_chat(db=db, payload=payload)
    return TransformerChatResponse(
        provider=payload.provider,
        model=model,
        answer=answer,
        sources=sources,
        diagnostics=diagnostics,
    )


@router.post("/classify", response_model=TransformerClassifyResponse)
def classify(
    payload: TransformerClassifyRequest,
    _: object = Depends(get_current_user),
) -> TransformerClassifyResponse:
    model, items = transformer_classify(payload)
    return TransformerClassifyResponse(model=model, items=items)


@router.post("/cluster", response_model=TransformerClusterResponse)
def cluster(
    payload: TransformerClusterRequest,
    _: object = Depends(get_current_user),
) -> TransformerClusterResponse:
    model, assignments, groups = transformer_cluster(payload)
    return TransformerClusterResponse(
        model=model, assignments=assignments, groups=groups
    )


@router.post("/rag/analyze", response_model=TransformerRagAnalyzeResponse)
def rag_analyze(
    payload: TransformerRagAnalyzeRequest,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
) -> TransformerRagAnalyzeResponse:
    analysis, model, sources, diagnostics = transformer_rag_analyze(
        db=db, payload=payload
    )
    return TransformerRagAnalyzeResponse(
        topic=payload.topic,
        provider=payload.provider,
        model=model,
        analysis=analysis,
        sources=sources,
        diagnostics=diagnostics,
    )


@router.post("/eval/run", response_model=TransformerEvalResponse)
def eval_run(
    payload: TransformerEvalRequest,
    _: object = Depends(get_current_user),
) -> TransformerEvalResponse:
    model, total, correct, accuracy, _ = transformer_eval(payload)
    return TransformerEvalResponse(
        model=model,
        total=total,
        correct=correct,
        accuracy=round(accuracy, 6),
    )


@router.post("/topics/quick-test", response_model=TransformerQuickTestResponse)
def topics_quick_test(
    payload: TransformerQuickTestRequest,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
) -> TransformerQuickTestResponse:
    model, total_topics, pass_count, average_score, items = quick_test_topics(
        db=db,
        payload=payload,
    )
    return TransformerQuickTestResponse(
        provider=payload.provider,
        model=model,
        total_topics=total_topics,
        pass_count=pass_count,
        average_score=round(float(average_score), 2),
        items=items,
    )


@router.post("/topics/quick-test/export")
def topics_quick_test_export(
    payload: TransformerQuickTestRequest,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
) -> StreamingResponse:
    model, total_topics, pass_count, average_score, items = quick_test_topics(
        db=db,
        payload=payload,
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "provider",
            "model",
            "total_topics",
            "pass_count",
            "average_score",
        ]
    )
    writer.writerow(
        [
            payload.provider,
            model,
            total_topics,
            pass_count,
            round(float(average_score), 2),
        ]
    )
    writer.writerow([])
    writer.writerow(
        [
            "code",
            "title",
            "score",
            "passed",
            "keyword_hits",
            "total_keywords",
            "retrieved_chunks",
            "mode",
            "latency_ms",
            "answer_preview",
        ]
    )
    for item in items:
        writer.writerow(
            [
                item.code,
                item.title,
                item.score,
                item.passed,
                item.keyword_hits,
                item.total_keywords,
                item.retrieved_chunks,
                item.mode,
                item.latency_ms,
                item.answer_preview,
            ]
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=transformer_quick_test.csv"
        },
    )


@router.post("/topics/quick-test/report-markdown")
def topics_quick_test_report_markdown(
    payload: TransformerQuickTestRequest,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_user),
) -> StreamingResponse:
    model, total_topics, pass_count, average_score, items = quick_test_topics(
        db=db,
        payload=payload,
    )
    content = build_quick_test_markdown_report(
        provider=payload.provider,
        model=model,
        total_topics=total_topics,
        pass_count=pass_count,
        average_score=average_score,
        items=items,
    )
    return StreamingResponse(
        iter([content]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=transformer_quick_test_report.md"
        },
    )
