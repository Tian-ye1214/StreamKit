import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING

st.markdown("""
<style>
    .stChatInputContainer {
        height: 100px !important;
    }
    .stChatInputContainer textarea {
        height: 100px !important;
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)


def initialization():
    if 'selected_alert' not in st.session_state:
        st.session_state.selected_alert = None
        st.session_state.news_selected_alert = None
    if 'alert_details' not in st.session_state:
        st.session_state.alert_details = {}
        st.session_state.news_alert_details = {}
    if 'province_alerts' not in st.session_state:
        st.session_state.province_alerts = None
        st.session_state.news_province_alerts = None
    if 'current_province' not in st.session_state:
        st.session_state.current_province = None
        st.session_state.news_current_province = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "deepseek-chat"


class WeatherAlertCrawler:
    def __init__(self):
        self.url = 'http://www.nmc.cn/rest/findAlarm?pageNo=1&pageSize=1000&signaltype=&signallevel=&province=&_='
        self.main_url = 'http://www.nmc.cn/publish/alarm/'
        self.suffix = '.html'
        self.alerts_dict = {}

    def load_existing_data(self):
        return list(self.alerts_dict.values())

    def get_data(self):
        try:
            str_html = requests.get(self.url)
            str_html.encoding = str_html.apparent_encoding
            soup = BeautifulSoup(str_html.text, 'html5lib')

            d_body = str(soup.body)
            d_body_dict = d_body.replace('<body>', '').replace('</body>', '')
            d_body_json = json.loads(d_body_dict)

            alerts = d_body_json['data']['page']['list']
            current_date = datetime.now().strftime('%Y/%m/%d')

            # 处理新的预警信息
            new_alerts = []
            for alert in alerts:
                alert_date = alert['issuetime'].split(' ')[0]
                # 只处理当天的预警
                if alert_date == current_date:
                    # 只添加新的预警
                    if alert['alertid'] not in self.alerts_dict:
                        alert['url'] = self.main_url + alert['alertid'] + self.suffix
                        # 将新预警添加到字典中
                        self.alerts_dict[alert['alertid']] = alert
                        new_alerts.append(alert)

            return new_alerts

        except Exception as e:
            st.error(f"获取数据时发生错误：{str(e)}")
            return []

    def update_data(self):
        """立即更新预警信息"""
        msg_counter = st.empty()
        msg_counter.info("正在更新预警信息...")
        self.get_data()
        msg_counter.success("更新预警信息完毕！")

    def user_query(self, province):
        # 先更新数据
        self.update_data()

        # 读取当前数据
        alerts = self.load_existing_data()

        # 如果用户输入"全部区域"，统计所有省份/直辖市的预警信息
        if province == "全部区域":
            provinces = set()
            for alert in alerts:
                title = alert['title']
                # 更新后的省份和直辖市列表
                province_list = [
                    # 直辖市
                    '北京市', '上海市', '天津市', '重庆市',
                    # 东北地区
                    '黑龙江省', '吉林省', '辽宁省',
                    # 自治区
                    '内蒙古自治区', '新疆维吾尔自治区', '西藏自治区',
                    '宁夏回族自治区', '广西壮族自治区',
                    # 华北地区
                    '河北省', '山西省',
                    # 西北地区
                    '陕西省', '青海省', '甘肃省',
                    # 华东地区
                    '山东省', '江苏省', '浙江省', '安徽省',
                    '福建省', '江西省',
                    # 华中地区
                    '河南省', '湖北省', '湖南省',
                    # 华南地区
                    '广东省', '海南省',
                    # 西南地区
                    '四川省', '贵州省', '云南省'
                ]

                # 同时处理简称和全称
                province_short_names = {
                    '内蒙古': '内蒙古自治区',
                    '新疆': '新疆维吾尔自治区',
                    '西藏': '西藏自治区',
                    '宁夏': '宁夏回族自治区',
                    '广西': '广西壮族自治区'
                }

                # 检查完整名称
                for p in province_list:
                    if p in title:
                        provinces.add(p)
                        break

                # 检查简称
                for short_name, full_name in province_short_names.items():
                    if short_name in title:
                        provinces.add(full_name)
                        break

            if provinces:
                st.info(f"\n当前共有 {len(provinces)} 个省份/直辖市发布了预警信息：")
                # 按地区分类显示
                regions = {
                    '直辖市': ['北京市', '上海市', '天津市', '重庆市'],
                    '东北地区': ['黑龙江省', '吉林省', '辽宁省'],
                    '华北地区': ['河北省', '山西省', '内蒙古自治区'],
                    '西北地区': ['陕西省', '青海省', '甘肃省', '宁夏回族自治区', '新疆维吾尔自治区'],
                    '华东地区': ['山东省', '江苏省', '浙江省', '安徽省', '福建省', '江西省'],
                    '华中地区': ['河南省', '湖北省', '湖南省'],
                    '华南地区': ['广东省', '广西壮族自治区', '海南省'],
                    '西南地区': ['四川省', '贵州省', '云南省', '西藏自治区']
                }

                for region, region_provinces in regions.items():
                    region_alerts = provinces.intersection(region_provinces)
                    if region_alerts:
                        st.markdown(f"\n{region}：")
                        for p in sorted(region_alerts):
                            st.markdown(f"- {p}")
            else:
                st.info("\n当前没有任何预警信息")
            return

        # 筛选指定省份的预警
        province_alerts = []
        for alert in alerts:
            if province in alert['title']:
                province_alerts.append(alert)

        return province_alerts

    def get_alert_detail(self, url):
        try:
            response = requests.get(url)
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html5lib')

            # 查找预警详情内容
            detail_div = soup.find('div', id='alarmtext')
            if detail_div:
                content = detail_div.get_text(strip=True)
                return content
            else:
                return "无法获取预警详细信息"
        except Exception as e:
            return f"获取详情时发生错误：{str(e)}"


class WeatherAlertNewsWriter:
    def __init__(self):
        self.defense_guides = self.load_defense_guides()
        API_SECRET_KEY = os.environ.get('ZhiZz_API_KEY')
        BASE_URL = os.environ.get('ZhiZz_URL')
        Model = st.session_state.get("selected_model", "deepseek-chat")
        self.llm = ChatOpenAI(
            model=Model,
            base_url=BASE_URL,
            api_key=API_SECRET_KEY,
            temperature=0,
        )
        self.news_template = """
        你是气象灾害预警领域的专家，你的任务是根据以下预警信息与防御指南生成与新闻模板形式一致的气象灾害预警新闻。
        预警信息:{alert_info}
        防御指南：{defense_guide}
        新闻模板：{news_template}
        注意事项：
        1. 生成内容必须基于给定的预警信息和防御指南；
        2. 生成内容必须符合新闻模板的形式，但不要参照其内容；
        3. 不要产生无关内容和虚假信息；
        """

    def load_defense_guides(self):
        """加载防御指南JSON文件"""
        guides = {}
        try:
            with open('./pages/SkySentry/defense_chunks.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    key = (item['地区'], item['预警信号'])
                    guides[key] = item['防御指南']
            return guides
        except Exception as e:
            st.error(f"加载预警数据库失败: {str(e)}")
            return {}

    def get_defense_guide(self, region, signal_type):
        """获取对应的防御指南"""
        # 先尝试匹配具体地区
        key = (region, signal_type)
        if key in self.defense_guides:
            return self.defense_guides[key]

        # 如果找不到具体地区的指南，使用中国气象局的通用指南
        key = ('中国气象局', signal_type)
        if key in self.defense_guides:
            return self.defense_guides[key]

        return "暂无对应的防御指南"

    def generate_news(self, alert_info, defense_guide, custom_template=None):
        """使用指定模板或默认模板生成预警新闻"""
        DEFAULT_NEWS_TEMPLATE = """
        四川省阿坝藏族羌族自治州若尔盖县气象台发布雷电黄色预警信号
        若尔盖县气象台2025年04月25日18时03分发布雷电黄色预警信号：若尔盖县达扎寺镇、唐克镇、辖曼镇、嫩哇乡、麦溪乡、红星镇、降扎乡、占哇乡、铁布镇、阿西镇、巴西镇、包座乡、求吉乡6小时内可能发生雷电活动，可能会造成雷电灾害。过程来临时可能伴有冰雹、阵性大风和短时强降水等强对流天气。请相关部门做好防雷工作，避免户外活动。
        防御指南：
        1.政府及相关部门按照职责做好防雷工作；
        2.密切关注天气，尽量避免户外活动。
        """
        try:
            template_content = custom_template if custom_template else DEFAULT_NEWS_TEMPLATE
            prompt = ChatPromptTemplate.from_template(self.news_template)
            messages = prompt.format_messages(
                alert_info=alert_info,
                defense_guide=defense_guide,
                news_template=template_content
            )
            response = self.llm(messages)
            return response.content
        except Exception as e:
            st.error(f"生成新闻时发生错误：{str(e)}")
            return None

    def analyze_alerts(self, alerts):
        """分析全国预警信息"""
        analysis_prompt = """
        请分析以下气象灾害预警信息，总结当前全国预警情况：
        1. 重点关注哪些地区出现预警？
        2. 主要出现了哪些类型的预警？
        3. 预警等级分布如何？
        4. 需要特别注意的灾害风险有哪些？
        5. 提供表格或具体统计分析。
        6. 以markdown格式返回。

        预警信息：
        {alerts_info}
        """
        alerts_summary = []
        for alert in alerts:
            alerts_summary.append(
                f"地区：{alert['title'].split('气象台')[0]}, 预警：{alert['title'].split('发布')[1].strip()}")

        messages = ChatPromptTemplate.from_template(analysis_prompt).format_messages(
            alerts_info="\n".join(alerts_summary)
        )
        response = self.llm(messages)
        return response.content


class WeatherAlertSystem(WeatherAlertCrawler):
    def __init__(self):
        super().__init__()
        self.news_writer = WeatherAlertNewsWriter()


def interaction(system):
    province = st.chat_input("请输入要查询的省份或者直辖市（例如：四川省、北京市）或输入'全部区域'查询全国预警信息",
                             key='query_input')

    if province:
        st.session_state.current_province = province
        st.session_state.province_alerts = system.user_query(province)
        st.session_state.selected_alert = None

    if st.session_state.current_province and not st.session_state.province_alerts:
        st.error(f"\n未找到{st.session_state.current_province}预警信息，请输入省份或直辖市")
        return

    if st.session_state.province_alerts:
        st.success(f"\n共找到 {len(st.session_state.province_alerts)} 条{st.session_state.current_province}预警信息：")
        col1, col2, col3 = st.columns(3)
        alerts_per_column = len(st.session_state.province_alerts) // 3
        remainder = len(st.session_state.province_alerts) % 3
        start_idx = 0
        for i, col in enumerate([col1, col2, col3]):
            with col:
                current_alerts = alerts_per_column + (1 if i < remainder else 0)
                for j in range(current_alerts):
                    if start_idx + j < len(st.session_state.province_alerts):
                        alert = st.session_state.province_alerts[start_idx + j]
                        alert_id = f"alert_{start_idx + j + 1}"
                        if st.button(f"{start_idx + j + 1}. {alert['title']}", key=alert_id):
                            st.session_state.selected_alert = alert_id
                            if alert_id not in st.session_state.alert_details:
                                st.session_state.alert_details[alert_id] = system.get_alert_detail(alert['url'])
                start_idx += current_alerts

        if st.session_state.selected_alert:
            st.info("\n预警详细信息：")
            st.markdown(st.session_state.alert_details[st.session_state.selected_alert])


def news_generation(system, use_custom_template):
    province = st.chat_input("请输入要查询的省份或者直辖市（例如：四川省、北京市）",
                             key='new_generation')

    if province:
        st.session_state.news_current_province = province
        st.session_state.news_province_alerts = system.user_query(province)
        st.session_state.news_selected_alert = None

    if not st.session_state.news_province_alerts and st.session_state.news_current_province:
        st.error(f"\n未找到{st.session_state.news_current_province}预警信息，请输入省份或直辖市")
        return

    if st.session_state.news_province_alerts:
        st.success(
            f"\n共找到 {len(st.session_state.news_province_alerts)} 条{st.session_state.news_current_province}预警信息：")
        col1, col2, col3 = st.columns(3)
        alerts_per_column = len(st.session_state.news_province_alerts) // 3
        remainder = len(st.session_state.news_province_alerts) % 3
        start_idx = 0
        for i, col in enumerate([col1, col2, col3]):
            with col:
                current_alerts = alerts_per_column + (1 if i < remainder else 0)
                for j in range(current_alerts):
                    if start_idx + j < len(st.session_state.news_province_alerts):
                        alert = st.session_state.news_province_alerts[start_idx + j]
                        alert_id = f"news_alert_{start_idx + j + 1}"
                        if st.button(f"{start_idx + j + 1}. {alert['title']}", key=alert_id):
                            st.session_state.news_selected_alert = alert_id
                            if alert_id not in st.session_state.news_alert_details:
                                st.session_state.news_alert_details[alert_id] = system.get_alert_detail(alert['url'])
                start_idx += current_alerts
    if st.session_state.news_selected_alert:
        selected_alert_index = int(st.session_state.news_selected_alert.split('_')[-1]) - 1
        if 0 <= selected_alert_index < len(st.session_state.news_province_alerts):
            selected_alert = st.session_state.news_province_alerts[selected_alert_index]
            title = selected_alert['title']
            alert_type = title.split('发布')[1].strip()

            defense_guide = system.news_writer.get_defense_guide(province, alert_type)
            st.info("\n防御指南：")
            st.markdown(defense_guide)

            news = system.news_writer.generate_news(
                st.session_state.news_alert_details[st.session_state.news_selected_alert],
                defense_guide, use_custom_template)
            st.markdown("\n生成的预警新闻：")
            st.markdown(news)


def main():
    initialization()
    st.markdown("""
    <h1 style='text-align: center;'>
        天眸预警 -- 实时天气预警查询
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        model_names = list(HIGHSPEED_MODEL_MAPPING.keys())
        selected_model_name = st.selectbox(
            "选择模型",
            options=model_names,
            index=1
        )
        st.session_state.selected_model = HIGHSPEED_MODEL_MAPPING[selected_model_name]
        st.markdown("联系作者")
        st.markdown(f"""
        👋🏼 [mwx66](https://github.com/mwx66)
        """, unsafe_allow_html=True)

    with st.expander("使用说明", expanded=False):
        st.markdown("""
        🌟 **欢迎使用天眸预警系统** 🌟
        
        💡 **快速上手**
        1. 在侧边栏选择心仪模型
        2. 选择功能标签页（查询预警信息/总结预警信息/撰写预警新闻）
        3. 输入省份或直辖市名称（如：四川省、北京市）
        4. 点击预警信息查看详情
        
        🎨 **功能特点**\n
        ✅ 实时获取全国气象预警信息<br>
        ✅ 按省份筛选预警信息<br>
        ✅ 查看预警详细信息<br>
        ✅ 获取相关防御指南<br>
        ✅ 自动生成预警新闻<br>
        ✅ 分析全国预警情况<br>

        <div style="background: #FCF3CF; padding: 15px; border-radius: 5px; margin-top: 20px;">
            🎭 试试这些功能：<br>
            • 输入"全部区域"查看全国预警情况<br>
            • 点击"分析全国预警信息"获取统计报告<br>
            • 在"撰写预警新闻"标签页中自定义新闻模板<br>
            每一次使用都是对气象灾害的及时预警！
        </div>
        
        <div style="margin-top: 20px; text-align: center;">
            <p>作者：<a href="https://github.com/mwx66" target="_blank">@mwx66</a></p>
            <p>项目地址：<a href="https://github.com/Tian-ye1214/StreamlitKit" target="_blank">@StreamKit</a></p>
        </div>
        """, unsafe_allow_html=True)

    system = WeatherAlertSystem()
    tab1, tab2, tab3 = st.tabs(['查询预警信息', '总结预警信息', '撰写预警新闻'])
    with tab1:
        interaction(system)

    with tab2:
        if st.button("分析全国预警信息"):
            system.update_data()
            alerts = system.load_existing_data()
            if alerts:
                st.success(f"\n共获取到 {len(alerts)} 条预警信息")
                st.info("\n正在分析预警信息...")
                analysis = system.news_writer.analyze_alerts(alerts)
                st.info("\n预警信息分析结果：")
                st.markdown(analysis)
            else:
                st.info("暂无预警信息")

    with tab3:
        use_custom_template = st.text_area("使用自定义模板", value=None)
        news_generation(system, use_custom_template)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'SkySentry'
current_page = 'SkySentry'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    initialization()
    st.session_state.previous_page = current_page
main()
