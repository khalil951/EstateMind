import re
import json
import time
import random
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PropertyListing:
	source: str
	title: Optional[str]
	price: Optional[float]
	governorate: Optional[str]
	city: Optional[str]
	property_type: Optional[str]
	surface_area: Optional[int]
	url: Optional[str]
	description: Optional[str]


class RequestManager:
	"""Handles requests with UA rotation, exponential backoff and proxy placeholders."""

	DEFAULT_UAS = [
		# A short list of common UAs; expand as needed
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
	]

	def __init__(self, user_agents: Optional[List[str]] = None, proxies: Optional[List[str]] = None, max_retries: int = 4, backoff_factor: float = 1.0):
		self.user_agents = user_agents or self.DEFAULT_UAS
		self.proxies = proxies or []  # placeholders like 'http://user:pass@host:port'
		self.max_retries = max_retries
		self.backoff_factor = backoff_factor
		self.session = requests.Session()

	def _choose_headers(self) -> Dict[str, str]:
		ua = random.choice(self.user_agents)
		return {"User-Agent": ua, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}

	def _choose_proxy(self) -> Optional[Dict[str, str]]:
		if not self.proxies:
			return None
		p = random.choice(self.proxies)
		return {"http": p, "https": p}

	def get(self, url: str, timeout: int = 20, allow_redirects: bool = True, **kwargs) -> requests.Response:
		last_exc = None
		for attempt in range(1, self.max_retries + 1):
			headers = self._choose_headers()
			proxy = self._choose_proxy()
			try:
				resp = self.session.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects, proxies=proxy, **kwargs)
				resp.raise_for_status()
				return resp
			except Exception as exc:
				last_exc = exc
				wait = self.backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 1)
				logger.warning("Request failed (%s). Retry %s/%s after %.1fs", url, attempt, self.max_retries, wait)
				time.sleep(wait)
		logger.error("All retries failed for %s: %s", url, last_exc)
		raise last_exc


class BaseScraper:
	"""Abstract base scraper providing JSONL incremental persistence."""

	def __init__(self, request_manager: RequestManager, output_path: Optional[str] = None, source: Optional[str] = None):
		self.request_manager = request_manager
		self.source = source or self.__class__.__name__
		self.output_path = Path(output_path or Path.cwd() / f"{self.source}.jsonl")
		self.output_path.parent.mkdir(parents=True, exist_ok=True)

	def save_listing(self, listing: PropertyListing):
		# Append a single normalized listing as JSONL to avoid data loss
		with self.output_path.open("a", encoding="utf-8") as f:
			json_line = json.dumps(asdict(listing), ensure_ascii=False)
			f.write(json_line + "\n")

	def scrape(self) -> List[PropertyListing]:
		"""Override in subclasses. Should return list of PropertyListing objects and call `save_listing` incrementally."""
		raise NotImplementedError()


def normalize_tunisian_data(listing: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize textual Tunisian data (governorate names, currency, surface, property type).

	Accepts a partial mapping (e.g., scraped dict) and returns normalized dict suitable for PropertyListing.
	"""
	# Arabic -> French mapping (extend as needed)
	ar_to_fr = {
		"تونس": "Tunis",
		"صفاقس": "Sfax",
		"نابل": "Nabeul",
		"سوسة": "Sousse",
	}

	result = dict(listing)  # shallow copy

	# Governorate normalization
	gov = result.get("governorate") or result.get("governorat") or result.get("location")
	if gov:
		gov = gov.strip()
		if gov in ar_to_fr:
			result["governorate"] = ar_to_fr[gov]
		else:
			result["governorate"] = gov

	# Currency normalization
	price_raw = result.get("price_raw") or result.get("price")
	price_val: Optional[float] = None
	if isinstance(price_raw, (int, float)):
		price_val = float(price_raw)
	elif isinstance(price_raw, str):
		s = price_raw.replace("\xa0", " ")
		s = s.replace(",", "")
		# detect 'Mille' or 'MD' meaning thousands
		if re.search(r"\bMille\b", s, re.I) or re.search(r"\bMD\b", s, re.I):
			num = re.search(r"(\d+[\d\s]*)", s)
			if num:
				n = int(re.sub(r"\s+", "", num.group(1)))
				price_val = float(n * 1000)
		else:
			# DT / TND or bare number
			num = re.search(r"(\d+[\d\s]*)", s)
			if num:
				n = int(re.sub(r"\s+", "", num.group(1)))
				price_val = float(n)
	result["price"] = price_val

	# Surface normalization: try surface field, otherwise regex on description
	surf = result.get("surface")
	if surf:
		try:
			result["surface_area"] = int(re.sub(r"\D", "", str(surf)))
		except Exception:
			result["surface_area"] = None
	else:
		desc = (result.get("description") or "")
		m = re.search(r"(\d{2,4})\s?m\s?\b|(\d{2,4})\s?m\u00B2", desc, re.I)
		if m:
			for g in m.groups():
				if g:
					result["surface_area"] = int(g)
					break
		else:
			result["surface_area"] = None

	# Property type mapping
	ptype = (result.get("property_type") or "").lower()
	if any(k in ptype for k in ["terrain", "terrain", "land"]):
		result["property_type"] = "Terrain"
	elif any(k in ptype for k in ["maison", "villa", "house"]):
		result["property_type"] = "Maison"
	elif any(k in ptype for k in ["appartement", "appart", "apt", "apartment"]):
		result["property_type"] = "Appartement"
	elif any(k in ptype for k in ["commercial", "commerce", "bureau"]):
		result["property_type"] = "Commercial"
	else:
		result["property_type"] = None

	# Location split: if combined like 'Tunis, La Marsa'
	loc = result.get("location")
	if loc and not result.get("city") and not result.get("governorate"):
		parts = [p.strip() for p in loc.split(",") if p.strip()]
		if len(parts) >= 2:
			result["city"] = parts[-1]
			result["governorate"] = parts[0]

	return result


class TayaraScraper(BaseScraper):
	"""Scraper for Tayara.tn (React SPA) using Playwright sync API."""

	START_URL = "https://www.tayara.tn/c/immobilier"

	def scrape(self) -> List[PropertyListing]:
		listings: List[PropertyListing] = []
		try:
			from playwright.sync_api import sync_playwright

			with sync_playwright() as pw:
				browser = pw.chromium.launch(headless=True)
				page = browser.new_page()
				page.goto(self.START_URL, timeout=30000)
				# Wait for grid to load
				try:
					page.wait_for_selector("[role=grid], .listing, .results", timeout=15000)
				except Exception:
					logger.info("Grid selector not found early; proceeding to attempt parse")

				# Infinite scroll until we have at least 50 unique links or max attempts
				hrefs = set()
				attempts = 0
				while len(hrefs) < 50 and attempts < 20:
					page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
					time.sleep(1.5 + random.random())
					# collect anchors
					anchors = page.query_selector_all("a[href*='/annonce/'], a[href*='/ad/'], a[href*='/announces/']")
					for a in anchors:
						try:
							h = a.get_attribute("href")
							if h:
								if h.startswith("/"):
									h = page.url.rstrip("/") + h
								hrefs.add(h)
						except Exception:
							continue
					attempts += 1

				# Limit to first 50
				href_list = list(hrefs)[:50]
				for href in href_list:
					try:
						page.goto(href, timeout=20000)
						page.wait_for_timeout(800)
						title = None
						price = None
						location_text = None
						description = None
						# try candidate selectors
						for sel in ["h1", ".title", ".ad-title"]:
							try:
								el = page.query_selector(sel)
								if el:
									title = el.inner_text().strip()
									break
							except Exception:
								continue

						# Price extraction: look for 'Prix' or digits with TND/DT/MD
						content = page.content()
						# Prix non spécifié handling
						if re.search(r"Prix non sp[eé]cifi", content, re.I):
							price = None
						else:
							m = re.search(r"(\d+[\d\s,]*)\s*(DT|TND|MD|\bMille\b)?", content)
							if m:
								n = int(re.sub(r"\D", "", m.group(1)))
								unit = (m.group(2) or "").upper()
								if unit == "MD" or re.search(r"Mille", m.group(0), re.I):
									price = float(n * 1000)
								else:
									price = float(n)

						# Location
						for sel in [".location", ".ad-location", ".city"]:
							try:
								el = page.query_selector(sel)
								if el:
									location_text = el.inner_text().strip()
									break
							except Exception:
								continue

						# Description
						for sel in [".description", ".ad-description", "#description"]:
							try:
								el = page.query_selector(sel)
								if el:
									description = el.inner_text().strip()
									break
							except Exception:
								continue

						gov = city = None
						if location_text:
							parts = [p.strip() for p in location_text.split(",") if p.strip()]
							if len(parts) >= 2:
								gov, city = parts[0], parts[-1]
							elif len(parts) == 1:
								gov = parts[0]

						normalized = normalize_tunisian_data({
							"price_raw": price,
							"location": location_text,
							"description": description,
							"property_type": None,
						})

						pl = PropertyListing(
							source=self.source,
							title=title,
							price=normalized.get("price"),
							governorate=normalized.get("governorate") or gov,
							city=normalized.get("city") or city,
							property_type=normalized.get("property_type"),
							surface_area=normalized.get("surface_area"),
							url=href,
							description=description,
						)
						self.save_listing(pl)
						listings.append(pl)
					except Exception as e:
						logger.exception("Failed to parse Tayara listing %s: %s", href, e)
				browser.close()
		except Exception as e:
			logger.exception("Playwright not available or Tayara scrape failed: %s", e)
		return listings


class MubawabScraper(BaseScraper):
	"""Scraper for Mubawab.tn using BeautifulSoup and pagination."""

	START_URL = "https://www.mubawab.tn/fr/listing-promotion:p:{page}"

	def scrape(self, max_pages: int = 5) -> List[PropertyListing]:
		listings: List[PropertyListing] = []
		for page_num in range(1, max_pages + 1):
			url = self.START_URL.format(page=page_num)
			try:
				resp = self.request_manager.get(url)
				soup = BeautifulSoup(resp.text, "lxml")
				items = soup.select("li.listingLi")
				if not items:
					logger.info("No items found on Mubawab page %s", page_num)
					break
				for it in items:
					try:
						a = it.find("a", href=True)
						href = a["href"] if a else None
						title = a.get_text(strip=True) if a else None
						price_el = it.select_one("span.priceTag")
						price_val = None
						if price_el:
							ptxt = price_el.get_text(strip=True)
							ptxt = ptxt.replace("DT", "").replace("TND", "").replace("\xa0", " ")
							ptxt = re.sub(r"[^0-9]", "", ptxt)
							if ptxt:
								price_val = float(ptxt)

						# attributes: rooms / surface inside feature icons
						attrs = it.select(".features, .listing-features, .icons")
						surface = None
						for acont in attrs:
							txt = acont.get_text(" ", strip=True)
							m = re.search(r"(\d{2,4})\s?m", txt, re.I)
							if m:
								surface = int(m.group(1))
								break

						# city/governorate guess from card
						loc = None
						loc_el = it.select_one(".listingLocality, .city")
						if loc_el:
							loc = loc_el.get_text(strip=True)

						normalized = normalize_tunisian_data({
							"price_raw": price_val,
							"surface": surface,
							"location": loc,
							"description": None,
						})

						pl = PropertyListing(
							source=self.source,
							title=title,
							price=normalized.get("price"),
							governorate=normalized.get("governorate"),
							city=normalized.get("city"),
							property_type=normalized.get("property_type"),
							surface_area=normalized.get("surface_area"),
							url=href,
							description=None,
						)
						self.save_listing(pl)
						listings.append(pl)
					except Exception:
						continue
                # Check for next page button; if not found, break loop
				# pagination: detect next button
				next_btn = soup.select_one("a.next, li.next a, .pagination .next")
				if not next_btn:
					break
			except Exception as e:
				logger.exception("Failed to parse Mubawab page %s: %s", page_num, e)
				break
		return listings


class TunisieAnnonceScraper(BaseScraper):
	"""Legacy table-based layout. Follows 'Détails' link for full description and surface."""

	START_URL = "http://www.tunisie-annonce.com/AnnoncesImmobilier.asp"

	def scrape(self, max_pages: int = 3) -> List[PropertyListing]:
		listings: List[PropertyListing] = []
		try:
			resp = self.request_manager.get(self.START_URL)
			soup = BeautifulSoup(resp.text, "lxml")
			table = soup.find("table")
			if not table:
				logger.info("No main table found on Tunisie Annonce")
				return listings
			rows = table.find_all("tr")
			for r in rows:
				cols = r.find_all("td")
				if len(cols) < 3:
					continue
				try:
					title = cols[0].get_text(strip=True)
					href = None
					link = cols[0].find("a", href=True)
					if link:
						href = link["href"]
						if href and href.startswith("/"):
							href = self.START_URL.rstrip("/") + href

					price = None
					price_txt = cols[1].get_text(strip=True)
					m = re.search(r"(\d+[\d\s]*)", price_txt)
					if m:
						price = float(re.sub(r"\s+", "", m.group(1)))

					description = None
					surface = None
					if href:
						try:
							det = self.request_manager.get(href)
							ds = BeautifulSoup(det.text, "lxml")
							desc_el = ds.find(text=re.compile(r"Description|D\w+tail", re.I))
							if desc_el:
								description = desc_el.parent.get_text(" ", strip=True)
							# attempt surface
							s = ds.get_text(" ", strip=True)
							m2 = re.search(r"(\d{2,4})\s?m\b", s, re.I)
							if m2:
								surface = int(m2.group(1))
						except Exception:
							logger.info("Failed following details link %s", href)

					normalized = normalize_tunisian_data({
						"price_raw": price,
						"surface": surface,
						"location": None,
						"description": description,
					})

					pl = PropertyListing(
						source=self.source,
						title=title,
						price=normalized.get("price"),
						governorate=normalized.get("governorate"),
						city=normalized.get("city"),
						property_type=normalized.get("property_type"),
						surface_area=normalized.get("surface_area"),
						url=href,
						description=description,
					)
					self.save_listing(pl)
					listings.append(pl)
				except Exception:
					continue
		except Exception as e:
			logger.exception("TunisieAnnonce scraping failed: %s", e)
		return listings


class TecnocasaScraper(BaseScraper):
	"""Basic stub for Tecnocasa; structure similar to other static scrapers."""

	def scrape(self) -> List[PropertyListing]:
		logger.info("TecnocasaScraper is a stub; implement selectors as needed")
		return []


def scraper_factory(name: str, request_manager: Optional[RequestManager] = None, output_path: Optional[str] = None) -> BaseScraper:
	rm = request_manager or RequestManager()
	name = name.lower()
	if "tayara" in name:
		return TayaraScraper(rm, output_path, source="Tayara")
	if "mubawab" in name:
		return MubawabScraper(rm, output_path, source="Mubawab")
	if "tunisie" in name or "annonce" in name:
		return TunisieAnnonceScraper(rm, output_path, source="TunisieAnnonce")
	if "tecnocasa" in name:
		return TecnocasaScraper(rm, output_path, source="Tecnocasa")
	raise ValueError(f"Unknown scraper name: {name}")

