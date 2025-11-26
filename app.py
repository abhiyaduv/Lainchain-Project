import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AI Content Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main { padding: 0rem 1rem; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        border: none;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1 { color: #667eea; font-weight: 700; }
    h2, h3 { color: #764ba2; font-weight: 600; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_local_model():
    """Load local HuggingFace model with error handling."""
    try:
        model_name = "distilgpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,
        )

        llm = HuggingFacePipeline(pipeline=text_gen)
        return llm, None

    except Exception as e:
        return None, f"Model loading error: {str(e)}"


def analyze_code_metrics(code_content: str):
    """Analyze Python code metrics."""
    try:
        lines = code_content.split("\n")

        total_lines = len(lines)
        code_lines = len(
            [l for l in lines if l.strip() and not l.strip().startswith("#")]
        )
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        blank_lines = total_lines - code_lines - comment_lines

        functions = len(re.findall(r"\bdef\s+\w+", code_content))
        classes = len(re.findall(r"\bclass\s+\w+", code_content))
        imports = len(
            re.findall(r"^\s*(?:import|from)\s+", code_content, re.MULTILINE)
        )
        docstrings = len(
            re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code_content)
        )

        comment_ratio = (comment_lines / total_lines * 100) if total_lines > 0 else 0

        complexity_score = functions * 2 + classes * 5
        if complexity_score < 10:
            complexity = "Low"
        elif complexity_score < 30:
            complexity = "Medium"
        else:
            complexity = "High"

        quality_score = min(
            100,
            (comment_ratio * 0.3)
            + (docstrings * 5)
            + (30 if functions > 0 else 0)
            + (20 if classes > 0 else 0)
            + (10 if imports > 0 else 0),
        )

        metrics = {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "docstrings": docstrings,
            "complexity": complexity,
            "comment_ratio": round(comment_ratio, 2),
            "quality_score": round(quality_score, 1),
        }

        return metrics, None

    except Exception as e:
        return None, f"Code analysis error: {str(e)}"


def analyze_text_patterns(text_content: str):
    """Analyze text content metrics."""
    try:
        words = [w for w in text_content.split() if w.strip()]
        sentences = [s.strip() for s in text_content.split(".") if s.strip()]
        paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

        word_count = len(words)
        unique_words = len(
            set(word.lower() for word in words if word.isalnum())
        )
        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0
        )

        sentence_count = len(sentences)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0
        )

        questions = text_content.count("?")
        exclamations = text_content.count("!")

        if avg_sentence_length < 15:
            readability = "Easy"
            readability_score = 85
        elif avg_sentence_length < 20:
            readability = "Moderate"
            readability_score = 65
        elif avg_sentence_length < 25:
            readability = "Challenging"
            readability_score = 50
        else:
            readability = "Complex"
            readability_score = 30

        vocab_richness = (
            unique_words / word_count * 100 if word_count > 0 else 0
        )

        patterns = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": len(paragraphs),
            "unique_words": unique_words,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "readability": readability,
            "readability_score": readability_score,
            "vocab_richness": round(vocab_richness, 1),
            "questions": questions,
            "exclamations": exclamations,
        }

        return patterns, None

    except Exception as e:
        return None, f"Text analysis error: {str(e)}"


def generate_insights_with_llm(llm, content_type: str, metrics: dict):
    """Generate AI insights with retries."""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            if content_type == "code":
                prompt_template = PromptTemplate(
                    input_variables=[
                        "functions",
                        "classes",
                        "complexity",
                        "quality",
                    ],
                    template=(
                        "Code Analysis: {functions} functions, "
                        "{classes} classes, {complexity} complexity, "
                        "{quality}% quality.\n\nProfessional Recommendations:\n1."
                    ),
                )

                chain = LLMChain(llm=llm, prompt=prompt_template)
                result = chain.run(
                    functions=metrics.get("functions", 0),
                    classes=metrics.get("classes", 0),
                    complexity=metrics.get("complexity", "Unknown"),
                    quality=metrics.get("quality_score", 0),
                )

            else:
                prompt_template = PromptTemplate(
                    input_variables=[
                        "words",
                        "sentences",
                        "readability",
                        "vocab",
                    ],
                    template=(
                        "Content Analysis: {words} words, {sentences} sentences, "
                        "{readability} readability, {vocab}% vocabulary richness.\n\n"
                        "Professional Recommendations:\n1."
                    ),
                )

                chain = LLMChain(llm=llm, prompt=prompt_template)
                result = chain.run(
                    words=metrics.get("word_count", 0),
                    sentences=metrics.get("sentence_count", 0),
                    readability=metrics.get("readability", "Unknown"),
                    vocab=metrics.get("vocab_richness", 0),
                )

            cleaned_result = result.strip()
            if len(cleaned_result) > 10:
                return cleaned_result, None
            else:
                retry_count += 1
                time.sleep(1)

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return None, f"AI generation failed after {max_retries} attempts: {str(e)}"
            time.sleep(2)

    return None, "Unable to generate insights. Please try again."


def create_quality_gauge(score: float, title: str):
    """Create gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20, "color": "#764ba2"}},
            delta={"reference": 70, "increasing": {"color": "#28a745"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                "bar": {"color": "#667eea"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 40], "color": "#ffebee"},
                    {"range": [40, 70], "color": "#fff9c4"},
                    {"range": [70, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={"family": "Arial"},
    )
    return fig


def main():
    st.markdown(
        """
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🎯 AI Content Analyzer Pro</h1>
        <p style='color: white; text-align: center; margin: 5px 0 0 0; font-size: 16px;'>Professional Analysis Powered by LangChain & Local AI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        analysis_type = st.radio(
            "Analysis Mode",
            ["🐍 Python Code", "📄 Text Content", "🔍 Auto-Detect"],
            help="Select the type of content to analyze",
        )

        st.markdown("---")

        st.markdown("### 📊 Features")
        st.markdown(
            """
        ✅ Local AI - No API Keys  
        ✅ Advanced Metrics  
        ✅ Quality Scoring  
        ✅ AI Recommendations  
        ✅ Professional Reports  
        """
        )

        st.markdown("---")

        st.markdown("### 🛠️ Tech Stack")
        st.markdown(
            """
        - LangChain  
        - HuggingFace (local)  
        - Streamlit  
        - Plotly  
        """
        )

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📤 Upload Content")

        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=["py", "txt", "md", "log", "java", "js", "cpp"],
            help="Supported: Python, Text, Markdown, Log, Java, JS, C++",
        )

        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")

                st.markdown(
                    f"""
                <div class='info-box'>
                    <strong>📁 File:</strong> {uploaded_file.name}<br>
                    <strong>📏 Size:</strong> {len(content)} characters<br>
                    <strong>📝 Lines:</strong> {len(content.splitlines())}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                with st.expander("👀 Preview Content", expanded=False):
                    st.code(
                        content[:1000] + ("..." if len(content) > 1000 else ""),
                        language="python" if uploaded_file.name.endswith(".py") else "text",
                    )

                analyze_button = st.button(
                    "🚀 Analyze Content", type="primary", use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                analyze_button = False
        else:
            st.markdown(
                """
            <div class='info-box'>
                <strong>👆 Get Started</strong><br>
                Upload a file to begin AI-powered analysis.
            </div>
            """,
                unsafe_allow_html=True,
            )
            analyze_button = False

    with col2:
        st.markdown("### 📊 Analysis Dashboard")

        if uploaded_file and analyze_button:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.markdown("🔄 Loading AI model...")
                progress_bar.progress(20)

                llm, model_error = load_local_model()
                if model_error:
                    st.error(f"❌ {model_error}")
                    return

                status_text.markdown("🔍 Detecting content type...")
                progress_bar.progress(40)
                time.sleep(0.3)

                is_code = uploaded_file.name.endswith(".py") or (
                    analysis_type == "🐍 Python Code"
                )

                status_text.markdown("⚙️ Analyzing content...")
                progress_bar.progress(60)

                if is_code:
                    metrics, analysis_error = analyze_code_metrics(content)
                else:
                    metrics, analysis_error = analyze_text_patterns(content)

                if analysis_error:
                    st.error(f"❌ {analysis_error}")
                    return

                progress_bar.progress(80)
                time.sleep(0.3)
                progress_bar.progress(100)
                time.sleep(0.3)
                progress_bar.empty()
                status_text.empty()

                st.markdown(
                    "### " + ("💻 Code Metrics" if is_code else "📝 Content Metrics")
                )

                if is_code:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("📄 Total Lines", metrics["total_lines"])
                    m2.metric("⚙️ Functions", metrics["functions"])
                    m3.metric("🏛️ Classes", metrics["classes"])
                    m4.metric("📦 Imports", metrics["imports"])

                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric("💬 Comments", f"{metrics['comment_ratio']}%")
                    n2.metric("📋 Docstrings", metrics["docstrings"])
                    n3.metric("🔧 Complexity", metrics["complexity"])
                    n4.metric("⭐ Quality", f"{metrics['quality_score']}/100")

                    st.plotly_chart(
                        create_quality_gauge(
                            metrics["quality_score"], "Code Quality Score"
                        ),
                        use_container_width=True,
                    )

                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("📝 Words", metrics["word_count"])
                    m2.metric("📄 Sentences", metrics["sentence_count"])
                    m3.metric("📑 Paragraphs", metrics["paragraph_count"])
                    m4.metric("🔤 Unique Words", metrics["unique_words"])

                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric("📏 Avg Word Length", f"{metrics['avg_word_length']}")
                    n2.metric("📊 Readability", metrics["readability"])
                    n3.metric("🎯 Vocabulary", f"{metrics['vocab_richness']}%")
                    n4.metric("❓ Questions", metrics["questions"])

                    st.plotly_chart(
                        create_quality_gauge(
                            metrics["readability_score"], "Readability Score"
                        ),
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("### 🤖 AI-Powered Recommendations")

                with st.spinner("Generating professional insights..."):
                    content_type = "code" if is_code else "text"
                    insights, insight_error = generate_insights_with_llm(
                        llm, content_type, metrics
                    )

                    if insight_error:
                        st.warning(f"⚠️ {insight_error}")
                        insights = (
                            "Manual Review Recommended: Analysis completed successfully. "
                            "Review the metrics above for improvement opportunities."
                        )

                    st.markdown(
                        f"""
                    <div class='success-box'>
                        <strong>💡 Professional Insights:</strong><br><br>
                        {insights}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("### 📄 Professional Report")

                report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                report = f"""
======================================================================
AI CONTENT ANALYZER PRO - PROFESSIONAL ANALYSIS REPORT
======================================================================

Generated: {report_timestamp}
File: {uploaded_file.name}
Type: {'Python Code Analysis' if is_code else 'Text Content Analysis'}
Analyzer: LangChain + HuggingFace DistilGPT2

======================================================================
ANALYSIS METRICS
======================================================================

"""

                for key, value in metrics.items():
                    report += f"{key.replace('_', ' ').title():<30} : {value}\n"

                report += f"""
======================================================================
AI-POWERED RECOMMENDATIONS
======================================================================

{insights}

======================================================================
METHODOLOGY
======================================================================

This report was generated using:
- LangChain framework for AI orchestration
- Local HuggingFace models (no external API calls)
- Pattern analysis and quality scoring

======================================================================
END OF REPORT
======================================================================

Report generated by AI Content Analyzer Pro
Powered by LangChain | 100% Local & Private
"""

                st.text_area("📋 Full Report", report, height=300)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        label="📥 Download Report (.txt)",
                        data=report,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        label="📊 Download Metrics (.csv)",
                        data="\n".join(
                            [f"{k},{v}" for k, v in metrics.items()]
                        ),
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.success("✅ Analysis completed successfully!")

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info(
                    "Tip: Try with a different file or restart the application."
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.markdown(
            """
        <div class='warning-box'>
            <strong>⚠️ Troubleshooting:</strong><br>
            1. Ensure all dependencies are installed<br>
            2. Use Python 3.8+<br>
            3. Restart the Streamlit server<br>
        </div>
        """,
            unsafe_allow_html=True,
        )
