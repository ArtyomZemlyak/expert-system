# sudo apt-get install libxml2-dev libxslt-dev python-dev
# pip install lxml

import os
from os import listdir
from os.path import isfile, join
from pprint import pprint
import time
import json

import requests
from lxml import html
from rich.progress import Progress

from HtmlParser import HtmlParser


CONFIG = {
    **json.loads(open("config.json", "r").read()),
    **json.loads(open("scripts/config.json", "r").read()),
}

URL = CONFIG["confluence_auth"]["url"]
KEYS = CONFIG["confluence_auth"]["spaces"]
AUTH_CONFL = (CONFIG["confluence_auth"]["user"], CONFIG["confluence_auth"]["pass"])

PATH_SAVE = CONFIG["save"]["path_save"]
SAVE_ALL_HREF = CONFIG["save"]["common"]["all_href"]


class ConfluencePageLoader:
    def __init__(
        self,
        url: str,
        auth: str,
        page_extract: str = '//*[@class="update-item-content"]/a/@href',
        more_link_extract: str = '//*[@class="more-link"]/@href',
    ) -> None:

        self.url = url
        self.auth = auth
        self.page_extract = page_extract
        self.more_link_extract = more_link_extract

    def load_pages_urls(self, keys: dict) -> list:

        href_list = []

        with Progress() as progress:
            task = progress.add_task(
                "[green]Pages links loading...", total=len(keys.keys())
            )

            for key, url_list_of_pages in keys.items():
                href_for_visit = f"{self.url}{url_list_of_pages}"

                while True:
                    r = requests.get(href_for_visit, auth=self.auth)
                    context = r.text
                    tree = html.fromstring(context)
                    current_href_list = tree.xpath(self.page_extract)

                    for current_href in current_href_list:
                        if current_href not in href_list:
                            href_list.append(current_href)

                    try:
                        href_for_visit = (
                            f"""{self.url}{tree.xpath(self.more_link_extract)[0]}"""
                        )
                    except IndexError:
                        break

                progress.update(task, advance=1)

        return href_list

    def to_dict(self, href_list: list) -> dict:

        print("Converting...")
        all_href_dict = {}
        all_href_dict["true"] = href_list
        all_href_dict["false"] = []
        all_href_dict["url"] = self.url

        return all_href_dict

    def get_all_text(self, all_href_dict: dict) -> str:
        texts = []
        url = all_href_dict["url"]

        with Progress() as progress:
            task = progress.add_task(
                "[green]Pages text loading...", total=len(all_href_dict["true"])
            )

            for href in all_href_dict["true"]:
                href_for_visit = url + href
                r = requests.get(href_for_visit, auth=self.auth)
                parser = HtmlParser(r.text)
                texts.append(parser.get_all_text())
                progress.update(task, advance=1)

        return "\n".join(texts)

    def save(self, all_href_dict: dict, path_save: str, file_name: str) -> None:

        print("Saving...")
        if all_href_dict == list:
            all_href_dict = self.to_dict(all_href_dict)

        if not os.path.exists(path_save):
            os.mkdir(path_save)

        json_file = open(os.path.join(path_save, file_name), "w", encoding="utf8")
        json_file.write(
            json.dumps(all_href_dict, indent=4, sort_keys=True, ensure_ascii=False)
        )
        json_file.close()


if __name__ == "__main__":

    URL = ""
    KEYS = {
        "": "",
    }

    start_time = time.time()

    confluence_page_loader = ConfluencePageLoader(url=URL, auth=AUTH_CONFL)
    href_list = confluence_page_loader.load_pages_urls(KEYS)
    all_href_dict = confluence_page_loader.to_dict(href_list)
    # confluence_page_loader.save(all_href_dict, PATH_SAVE, SAVE_ALL_HREF)
    all_text = confluence_page_loader.get_all_text(all_href_dict)

    with open("TEST/all_text_confl_mfc.txt", "w") as f:
        f.write(all_text)

    print("TOTAL TIME: ", (time.time() - start_time))
