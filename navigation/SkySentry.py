import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from openai import AsyncOpenAI
import streamlit as st
from pages.Functions.Constants import HIGHSPEED_MODEL_MAPPING
from pages.Functions.Prompt import SkySentry_prompt
import asyncio


async def initialization():
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
    if "Client" not in st.session_state:
        st.session_state.Client = AsyncOpenAI(api_key=os.environ.get('ZhiZz_API_KEY'), base_url=os.environ.get('ZhiZz_URL'))


class WeatherAlertCrawler:
    def __init__(self):
        self.url = 'http://www.nmc.cn/rest/findAlarm?pageNo=1&pageSize=1000&signaltype=&signallevel=&province=&_='
        self.main_url = 'http://www.nmc.cn/publish/alarm/'
        self.suffix = '.html'
        self.alerts_dict = {}

    async def load_existing_data(self):
        return list(self.alerts_dict.values())

    async def get_data(self):
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

    async def update_data(self):
        """立即更新预警信息"""
        msg_counter = st.empty()
        msg_counter.info("正在更新预警信息...")
        await self.get_data()
        msg_counter.success("更新预警信息完毕！")

    async def user_query(self, province):
        # 先更新数据
        await self.update_data()

        # 读取当前数据
        alerts = await self.load_existing_data()

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

    async def get_alert_detail(self, url):
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
        self.defense_guides = {}

    @classmethod
    async def create(cls):
        instance = cls()
        instance.defense_guides = await instance.load_defense_guides()
        return instance

    async def load_defense_guides(self):
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

    async def get_defense_guide(self, region, signal_type):
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

    async def generate_news(self, alert_info, defense_guide, custom_template=None):
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
            message = SkySentry_prompt(alert_info, defense_guide, template_content, generate_news=True)
            await self._call_llm(message)
        except Exception as e:
            st.error(f"生成新闻时发生错误：{str(e)}")
            return None

    async def analyze_alerts(self, alerts):
        alerts_summary = []
        for alert in alerts:
            alerts_summary.append(
                f"地区：{alert['title'].split('气象台')[0]}, 预警：{alert['title'].split('发布')[1].strip()}")

        messages = SkySentry_prompt(alerts_summary, None, None, generate_news=False)
        await self._call_llm(messages)

    async def _call_llm(self, messages):
        with st.spinner("正在生成回答..."):
            reason_placeholder = st.empty()
            message_placeholder = st.empty()
            content = ""
            reasoning_content = ""
            async for chunk in await st.session_state.Client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8192,
                    stream=True
            ):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if getattr(delta, 'reasoning_content', None):
                        reasoning_content += delta.reasoning_content
                        reason_placeholder.markdown(
                            f"<div style='background:#f0f0f0; border-radius:5px; padding:10px; margin-bottom:10px; font-size:14px;'>"
                            f"🤔 {reasoning_content}</div>",
                            unsafe_allow_html=True
                        )
                    if delta and delta.content is not None:
                        content += delta.content
                        message_placeholder.markdown(
                            f"<div style='font-size:16px; margin-top:10px;'>{content}</div>",
                            unsafe_allow_html=True
                        )


class WeatherAlertSystem(WeatherAlertCrawler):
    def __init__(self):
        super().__init__()
        self.news_writer = None

    @classmethod
    async def create(cls):
        instance = cls()
        instance.news_writer = await WeatherAlertNewsWriter.create()
        return instance


async def interaction(system):
    province = st.chat_input("请输入要查询的省份或者直辖市（例如：四川省、北京市）或输入'全部区域'查询全国预警信息",
                             key='query_input')

    if province:
        st.session_state.current_province = province
        st.session_state.province_alerts = await system.user_query(province)
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
                                st.session_state.alert_details[alert_id] = await system.get_alert_detail(alert['url'])
                start_idx += current_alerts

        if st.session_state.selected_alert:
            st.info("\n预警详细信息：")
            st.markdown(st.session_state.alert_details[st.session_state.selected_alert])


async def news_generation(system, use_custom_template):
    province = st.chat_input("请输入要查询的省份或者直辖市（例如：四川省、北京市）",
                             key='new_generation')

    if province:
        st.session_state.news_current_province = province
        st.session_state.news_province_alerts = await system.user_query(province)
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
                                st.session_state.news_alert_details[alert_id] = await system.get_alert_detail(alert['url'])
                start_idx += current_alerts
    if st.session_state.news_selected_alert:
        selected_alert_index = int(st.session_state.news_selected_alert.split('_')[-1]) - 1
        if 0 <= selected_alert_index < len(st.session_state.news_province_alerts):
            selected_alert = st.session_state.news_province_alerts[selected_alert_index]
            title = selected_alert['title']
            alert_type = title.split('发布')[1].strip()

            defense_guide = await system.news_writer.get_defense_guide(province, alert_type)
            st.info("\n防御指南：")
            st.markdown(defense_guide)

            await system.news_writer.generate_news(st.session_state.news_alert_details[st.session_state.news_selected_alert],
                                             defense_guide, use_custom_template)


async def main():
    await initialization()
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
            index=0
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

    system = await WeatherAlertSystem.create()
    tab1, tab2, tab3 = st.tabs(['查询预警信息', '总结预警信息', '撰写预警新闻'])
    with tab1:
        await interaction(system)

    with tab2:
        if st.button("分析全国预警信息"):
            await system.update_data()
            alerts = await system.load_existing_data()
            if alerts:
                st.success(f"\n共获取到 {len(alerts)} 条预警信息")
                st.info("\n正在分析预警信息...")
                await system.news_writer.analyze_alerts(alerts)
            else:
                st.info("暂无预警信息")

    with tab3:
        use_custom_template = st.text_area("使用自定义模板", value=None)
        await news_generation(system, use_custom_template)


if 'previous_page' not in st.session_state:
    st.session_state.previous_page = 'SkySentry'
current_page = 'SkySentry'
if current_page != st.session_state.previous_page:
    st.session_state.clear()
    st.session_state.previous_page = current_page
asyncio.run(main())
