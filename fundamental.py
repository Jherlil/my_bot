import feedparser
from datetime import datetime
from utils import log

class FundamentalAnalyzer:
    def __init__(self, buffer_minutes=60):
        self.buffer_minutes = buffer_minutes
        self.feed_url = "https://nfs.forexfactory.net/rss/economic_calendar.xml"

    def check_high_impact_news(self):
        log("Verificando notícias...")
        feed = feedparser.parse(self.feed_url)
        now = datetime.utcnow()
        for entry in feed.entries:
            impact = entry.get('category', '').lower()
            event_time = self._parse_time(entry.get('published', ''))
            if event_time and impact in ['high', 'important']:
                diff = (event_time - now).total_seconds() / 60
                if 0 <= diff <= self.buffer_minutes:
                    log(f"Notícia {entry.title} em {diff:.1f} min — Pausando robô!")
                    return True
        return False

    def _parse_time(self, time_str):
        try:
            return datetime.strptime(time_str, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            return None
