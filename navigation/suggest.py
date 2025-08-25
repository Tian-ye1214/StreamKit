import os
import json
from datetime import datetime
import streamlit as st


st.set_page_config(
    page_title="意见与建议",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
)


def get_safe_username(raw_username: str) -> str:
    if not raw_username:
        return "anonymous"
    safe = "".join([c for c in raw_username if c.isalnum() or '\u4e00' <= c <= '\u9fff'])
    safe = safe.strip()[:50]
    return safe or "anonymous"


def get_suggestions_dir(username: str) -> str:
    base_dir = os.path.join("user_suggestion", username, "suggestions")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_suggestion(username: str, title: str, content: str, contact: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    record = {
        "username": username,
        "title": title,
        "content": content,
        "contact": contact or "",
        "timestamp": ts,
        "iso_time": datetime.now().isoformat(),
    }
    per_file = os.path.join(get_suggestions_dir(username), f"suggest_{ts}.json")
    with open(per_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    aggregate_path = os.path.join(get_suggestions_dir(username), "suggestions_all.json")
    aggregate = []
    if os.path.exists(aggregate_path):
        try:
            with open(aggregate_path, "r", encoding="utf-8") as f:
                aggregate = json.load(f)
            if not isinstance(aggregate, list):
                aggregate = []
        except Exception:
            aggregate = []
    aggregate.append(record)
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    return per_file


def render_history(username: str):
    dir_path = get_suggestions_dir(username)
    aggregate_path = os.path.join(dir_path, "suggestions_all.json")
    if not os.path.exists(aggregate_path):
        st.info("暂无历史建议记录。")
        return
    try:
        with open(aggregate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            st.info("暂无历史建议记录。")
            return
        for item in reversed(data[-50:]):
            with st.expander(f"{item.get('title','(无标题)')} · {item.get('timestamp','')} "):
                st.markdown(f"- 提交人：{item.get('username','')}\n- 联系方式：{item.get('contact','')}\n- 提交时间：{item.get('iso_time','')}")
                st.write(item.get("content", ""))
    except Exception as e:
        st.error(f"读取历史记录失败：{e}")


def main():
    st.title("意见与建议 📝")
    st.caption("欢迎提出功能建议、体验问题或 bug 反馈。我们会认真查看并持续改进！")

    default_username = st.session_state.get("username", "")
    title = st.text_input("建议标题", placeholder="一句话概括您的建议或问题")
    content = st.text_area("建议详情", height=180, placeholder="请尽量描述清楚场景、期望与复现步骤等")
    username_input = st.text_input("您的昵称/用户名（可选）", value=default_username, placeholder="例如：张三")
    username = get_safe_username(username_input)
    if username and username != st.session_state.get("username"):
        st.session_state["username"] = username
    contact = st.text_input("联系方式（可选）", placeholder="邮箱/微信/电话，用于后续沟通")

    col1, col2 = st.columns([1, 1])
    with col1:
        submit = st.button("提交", type="primary")
    with col2:
        show_history = st.toggle("查看历史记录", value=False)

    if submit:
        if not title.strip() or not content.strip():
            st.warning("请填写完整的标题与详情后再提交。")
        else:
            try:
                file_path = save_suggestion(username, title.strip(), content.strip(), contact.strip())
                st.success("提交成功，感谢您的反馈！")
                st.toast(f"已保存到：{file_path}")
                st.balloons()
            except Exception as e:
                st.error(f"保存失败：{e}")

    if show_history:
        render_history(username)


if __name__ == "__main__":
    main()