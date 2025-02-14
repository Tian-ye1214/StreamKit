from duckduckgo_search import DDGS


class WebSearch:
    def __init__(self, query, region='cn-zh', max_results=3):
        """
         :param query: 用户提问
         :param region:检索地区，'cn-zh'为中国
         :param max_results: 最大检索数
         :return: list(key)
         """
        self.query = query
        self.region = region
        self.max_results = max_results

    def _format_result(self, results, keys_mapping):
        """统一格式化搜索结果"""
        formatted = []
        for item in results:
            new_item = {}
            for new_key, old_key in keys_mapping.items():
                new_item[new_key] = item.get(old_key, '')
            formatted.append(new_item)
        return formatted

    def text_search(self):
        with DDGS() as ddgs:
            search_result = [r for r in ddgs.text(self.query, region=self.region, max_results=self.max_results)]
        return self._format_result(search_result, {'title': 'title', 'href': 'href', 'body': 'body'})

    def news_search(self):
        with DDGS() as ddgs:
            search_result = [r for r in ddgs.news(self.query, region=self.region, max_results=self.max_results)]
        return self._format_result(search_result, {'title': 'title', 'href': 'url', 'body': 'body'})

    def image_search(self):
        with DDGS() as ddgs:
            search_result = [r for r in ddgs.images(self.query, region=self.region, max_results=self.max_results)]
        return self._format_result(search_result, {'title': 'title', 'href': 'url', 'body': 'description'})

    def video_search(self):
        with DDGS() as ddgs:
            search_result = [r for r in ddgs.videos(self.query, region=self.region, max_results=self.max_results)]
        return self._format_result(search_result, {'title': 'title', 'href': 'content', 'body': 'description'})




