import os
import re
from time import sleep

import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from urllib3.util import url
from webdriver_manager.chrome import ChromeDriverManager

browser = webdriver.Chrome(ChromeDriverManager().install())
output_path = '../data/master_files/'


def build_master_speech():
    # 1996 to 2005 [1996, 2006)

    master_list = []
    master_cols = ['Date', 'Title', 'Url', 'Speaker', 'Location']

    for year in range(1996, 2006):
        year_page_url = Request(headers={'User-Agent': 'Mozilla/5.0'},
                                url='https://www.federalreserve.gov/newsevents/' + 'speech' + '/' + str(
                                    year) + 'speech' + '.htm')
        response = urlopen(year_page_url)
        sleep(3)
        soup = bs(response, 'html.parser')
        response.close()

        ul_el = soup.find("ul", {"id": 'speech' + "Index"})
        ul_list = ul_el.findAll('li')

        for current_li in ul_list:
            EventDate = current_li.find(text=True, recursive=False)
            EventDate = EventDate.strip()
            EventDate = re.sub(r'\s+', ' ', EventDate)

            title_div = current_li.find("div", {"class": "title"})
            EventTitle = title_div.get_text()
            EventTitle = EventTitle.strip()
            EventTitle = re.sub(r'\s+', ' ', EventTitle)

            EventUrl = title_div.a['href']

            speaker_div = current_li.find("div", {"class": "speaker"})
            EventSpeaker = speaker_div.get_text()
            EventSpeaker = EventSpeaker.strip()
            EventSpeaker = re.sub(r'\s+', ' ', EventSpeaker)

            location_div = current_li.find("div", {"class": "location"})
            EventLocation = location_div.get_text()
            EventLocation = EventLocation.strip()
            EventLocation = re.sub(r'\s+', ' ', EventLocation)

            master_list.append([EventDate, EventTitle, EventUrl, EventSpeaker, EventLocation])

        page_url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
        page_count = 48

    browser.get(page_url)
    temp = []

    for page_number in range(page_count):
        print("Page: ", page_number + 1)
        page = browser.page_source
        soup = bs(page, 'html.parser')

        divs_for_events = soup.findAll("div", {"class": "row ng-scope"})

        for current_div in divs_for_events:
            print(current_div.get_text())
            datetime_div = current_div.find("div", {"class": "col-xs-3 col-md-2 eventlist__time"})
            EventDate = datetime_div.get_text()
            EventDate = EventDate.strip()
            EventDate = re.sub(r'\s+', ' ', EventDate)

            event_div = current_div.find("div", {"class": "col-xs-9 col-md-10 eventlist__event"})

            p_title = event_div.find("p", {"class": "itemTitle"})
            EventTitle = p_title.get_text()
            EventTitle = EventTitle.strip()
            EventTitle = re.sub(r'\s+', ' ', EventTitle)

            EventUrl = p_title.a['href']

            p_speaker = event_div.find("p", {"class": "news__speaker ng-binding"})
            EventSpeaker = p_speaker.get_text()
            EventSpeaker = EventSpeaker.strip()
            EventSpeaker = re.sub(r'\s+', ' ', EventSpeaker)

            p_location = event_div.find("p", {"class": "result__location ng-binding"})
            EventLocation = p_location.get_text()
            EventLocation = EventLocation.strip()
            EventLocation = re.sub(r'\s+', ' ', EventLocation)

            temp.append([EventDate, EventTitle, EventUrl, EventSpeaker, EventLocation])

        element = browser.find_element_by_xpath('//*[@id="article"]/ul[1]/li[11]/a')
        browser.execute_script("arguments[0].click();", element)
        sleep(3)

    browser.close()
    master_list.extend(temp[::-1])
    master_df = pd.DataFrame(master_list, columns=master_cols).drop_duplicates(inplace=False).reindex()
    master_df.to_csv(output_path + 'master_' + 'speech' + '.csv', index=False)

    return 0


def build_master_current_meeting():
    master_list = []
    master_cols = ['Year', 'Date', 'ItemNameCommon', 'ItemName', 'Url']

    year_page_url = Request(headers={'User-Agent': 'Mozilla/5.0'},
                            url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm')

    response = urlopen(year_page_url)
    sleep(3)
    soup = bs(response, 'html.parser')
    response.close()

    divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
    year = 2022
    for current_div in divs_for_cycles[0:]:
        non_shaded_divs = current_div.findAll("div", {"class": "row fomc-meeting"})
        shaded_divs = current_div.findAll("div", {"class": "fomc-meeting--shaded row fomc-meeting"})
        all_sub_divs = non_shaded_divs + shaded_divs
        print(len(all_sub_divs))

        for sub_div in all_sub_divs:
            month_divs = sub_div.find_all("div", {"class": "fomc-meeting__month"})
            day_divs = sub_div.find_all("div", {"class": "fomc-meeting__date"})

            month_text = month_divs[0].get_text().strip()
            day_text = day_divs[0].get_text().strip()

            date_text = month_text + ',' + day_text
            minutes_divs = sub_div.find_all("div", {"class": "fomc-meeting__minutes"})
            if (len(minutes_divs) > 0):
                minutes_div = minutes_divs[0]
                links = minutes_div.findAll('a')
                print(links)
                if (len(links) > 0):
                    ItemNameCommon = minutes_div.strong.get_text()

                    for link in links:
                        if ItemNameCommon == None:
                            ItemNameCommon = link.get_text()
                        if ItemNameCommon.isspace():
                            ItemNameCommon = link.get_text()
                        if link.get_text() == "HTML":
                            master_list.append([year, date_text, ItemNameCommon, link.get_text(), link['href']])

        year = year - 1
    #------------------------------------Pre 2016

    for year in range(2016, 1995, -1):
        year_page_url = Request(headers={'User-Agent': 'Mozilla/5.0'},
                                url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical' + str(year) + '.htm')
        response = urlopen(year_page_url)
        sleep(3)
        soup = bs(response, 'html.parser')
        response.close()

        divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
        if len(divs_for_cycles) < 1:
            divs_for_cycles = soup.findAll("div", {"class": "panel panel-default panel-padded"})

        for current_div in divs_for_cycles:
            h5_tag = current_div.find('h5')
            h5_tag_text = h5_tag.get_text()
            p_tags = current_div.findAll('p')
            for p_tag in p_tags:
                ItemNameCommon = p_tag.find(text=True, recursive=False)
                links = p_tag.findAll('a')
                print(links)
                for link in links:
                    if ItemNameCommon == None:
                        ItemNameCommon = link.get_text()
                    if ItemNameCommon.isspace():
                        ItemNameCommon = link.get_text()
                    if link.get_text() == "HTML":
                        master_list.append([year, h5_tag_text, ItemNameCommon, link.get_text(), link['href']])

    master_list = master_list[::-1]
    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv(output_path + 'master_mm.csv', index=False)


# build_master_speech()
build_master_current_meeting()
