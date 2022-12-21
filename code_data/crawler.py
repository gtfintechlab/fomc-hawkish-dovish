import pandas as pd
import string

import urllib.request as url
from bs4 import BeautifulSoup as bs
import re
import sys
from threading import Thread
from time import time
import os
from time import sleep
import shutil

from selenium import webdriver


def build_master(start_year=2010, end_year=2011):
    index = 0
    master_list = []
    master_cols = ['Year', 'Date', 'ItemNameCommon', 'ItemName', 'Url']

    for year in range(start_year, end_year):
        local_path = './data/' + str(year)
        if (not os.path.isdir(local_path)):
            os.mkdir(local_path)

        year_page_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical' + str(year) + '.htm'
        print(year_page_url)
        response = url.urlopen(year_page_url)
        sleep(3)
        soup = bs(response, 'html.parser')
        response.close()
        url.urlretrieve(year_page_url, local_path + '/' + 'yearpage.htm')

        divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
        if (len(divs_for_cycles) < 1):
            divs_for_cycles = soup.findAll("div", {"class": "panel panel-default panel-padded"})

        for current_div in divs_for_cycles:
            h5_tag = current_div.find('h5')
            h5_tag_text = h5_tag.get_text()
            p_tags = current_div.findAll('p')
            for p_tag in p_tags:
                ItemNameCommon = p_tag.find(text=True, recursive=False)
                # print(h5_tag, ItemNameCommon)
                links = p_tag.findAll('a')
                for link in links:
                    if ItemNameCommon == None:
                        ItemNameCommon = link.get_text()
                    if ItemNameCommon.isspace():
                        ItemNameCommon = link.get_text()
                    # print(ItemNameCommon, link.get_text())
                    master_list.append([year, h5_tag_text, ItemNameCommon, link.get_text(), link['href']])

        index = index + 1
    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master.csv', index=False)

    return 0


def download_files():
    start = 0
    end = 5
    master_df = pd.read_csv('master.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('LocalPath')
    res_list = []
    err_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name_short = row['Url'].split('/')

            file_name = 'Z:/FOMC/data/' + str(row['Year']) + '/' + file_name_short[-1]
            if (file_name_short[1] == 'boarddocs' and row['ItemName'] == 'Statement'):
                file_name = 'Z:/FOMC/data/' + str(row['Year']) + 'Statement_' + file_name_short[-2] + '.htm'
            # print(file_name)
            if (not os.path.isfile(file_name)):
                post_fix = row['Url']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                if (post_fix == 'http://www.minneapolisfed.org/bb/'):
                    continue
                if ('http' in post_fix):
                    curr_url = post_fix

                url.urlretrieve(curr_url, file_name)
                sleep(3)

            curr_row = list(row)
            curr_row.append(file_name)
            res_list.append(curr_row)

        except Exception as e:
            print(e)
            err_list.append(list(row))
    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('master_output.csv', index=False)
    err_df = pd.DataFrame(err_list, columns=column_list[:-1])
    err_df.to_csv('error.csv', index=False)


def clean_date():
    start = 0
    end = 50
    master_df = pd.read_csv('master_output.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('DateEventType')
    column_list.append('StartDate')
    column_list.append('EndDate')
    res_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)

        curr_row = list(row)

        date_string = row['Date']

        DateEventType = ""
        if "Meeting" in date_string:
            DateEventType = "Meeting"
        elif "Conference Call" in date_string:
            DateEventType = "Conference Call"
        elif "(unscheduled)" in date_string:
            DateEventType = "unscheduled"
        curr_row.append(DateEventType)

        # remove "Meeting - ", "Conference Call -",
        date_string = re.sub(r' Meeting - \d\d\d\d', '', date_string, flags=re.IGNORECASE)
        date_string = re.sub(r' Conference Call - \d\d\d\d', '', date_string, flags=re.IGNORECASE)
        date_string = re.sub(r' \(unscheduled\) - \d\d\d\d', '', date_string, flags=re.IGNORECASE)

        StartDate = date_string
        EndDate = date_string

        if ("-" not in date_string):  # Single date
            date_string_split = date_string.split(" ")
            StartMonth = date_string_split[0]
            EndMonth = date_string_split[0]
            StartDay = date_string_split[1]
            EndDay = date_string_split[1]
            StartDate = StartMonth + "/" + StartDay + "/" + str(row['Year'])
            EndDate = EndMonth + "/" + EndDay + "/" + str(row['Year'])
        else:
            date_string = re.sub(r'\s*-\s*', ',', date_string, flags=re.IGNORECASE)
            m = re.search(r'\d,\d', date_string, flags=re.IGNORECASE)
            if m:
                date_string_split = date_string.split(" ")
                StartMonth = date_string_split[0]
                EndMonth = date_string_split[0]
                date_string_split_further = date_string_split[1].split(",")
                StartDay = date_string_split_further[0]
                EndDay = date_string_split_further[1]
                StartDate = StartMonth + "/" + StartDay + "/" + str(row['Year'])
                EndDate = EndMonth + "/" + EndDay + "/" + str(row['Year'])
            else:
                date_string_split = date_string.split(",")
                date_string_split_further = date_string_split[0].split(" ")
                StartMonth = date_string_split_further[0]
                StartDay = date_string_split_further[1]
                date_string_split_further = date_string_split[1].split(" ")
                EndMonth = date_string_split_further[0]
                EndDay = date_string_split_further[1]

                StartDate = StartMonth + "/" + StartDay + "/" + str(row['Year'])
                EndDate = EndMonth + "/" + EndDay + "/" + str(row['Year'])

        curr_row.append(StartDate)
        curr_row.append(EndDate)

        res_list.append(curr_row)

    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('master_output_2.csv', index=False)
    return 0


def build_master_newsevents(category='speech'):
    # 1996 to 2005 [1996, 2006)

    master_list = []
    master_cols = ['Date', 'Title', 'Url', 'Speaker', 'Location']

    for year in range(1996, 2006):
        year_page_url = 'https://www.federalreserve.gov/newsevents/' + category + '/' + str(year) + category + '.htm'
        print(year_page_url)
        response = url.urlopen(year_page_url)
        sleep(3)
        soup = bs(response, 'html.parser')
        response.close()

        ul_el = soup.find("ul", {"id": category + "Index"})
        ul_list = ul_el.findAll('li')
        if (category == 'testimony' and year == 1997):
            ul_list = ul_list[:-1]

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

    # 2006 to today
    if (category == 'speech'):
        page_url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
        page_count = 43
    elif (category == 'testimony'):
        page_url = 'https://www.federalreserve.gov/newsevents/testimony.htm'
        page_count = 13
    browser = webdriver.Chrome()
    browser.get(page_url)

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

            master_list.append([EventDate, EventTitle, EventUrl, EventSpeaker, EventLocation])

        # element = browser.find_element_by_xpath('//*[@id="article"]/ul[1]/li[-1]/a')
        element = browser.find_element_by_link_text('Next')
        element.click()
        sleep(3)

    browser.close()

    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_' + category + '.csv', index=False)

    return 0


def build_master_conferences():
    # 1996 to 2005 [1996, 2006)

    master_list = []
    master_cols = ['Dates', 'Title', 'Url', 'ConferenceType', 'Location']

    page_url = 'https://www.federalreserve.gov/past-conferences.htm'

    '''
    response = url.urlopen(page_url)
    soup = bs(response, 'html.parser')
    response.close()
    '''
    f = open('temp.txt')
    response = f.read()
    f.close()
    soup = bs(response, 'html.parser')

    # list_div = soup.find("div", {"class": "row eventList"})
    divs_for_events = soup.findAll("div", {"class": "row"})
    print(len(divs_for_events))

    for current_div in divs_for_events:
        print(current_div.get_text())
        try:
            datetime_div = current_div.find("div", {"class": "col-xs-3 col-md-3 eventlist__time"})
            EventDate = datetime_div.get_text()
            EventDate = EventDate.strip()
            EventDate = re.sub(r'\s+', ' ', EventDate)

            event_div = current_div.find("div", {"class": "col-xs-9 col-md-9 eventlist__event"})

            p_tags = event_div.findAll("p")

            p_title = p_tags[0]
            EventTitle = p_title.get_text()
            EventTitle = EventTitle.strip()
            EventTitle = re.sub(r'\s+', ' ', EventTitle)

            EventUrl = p_title.a['href']

            ConferenceType = current_div['data-type']

            p_location = p_tags[1]
            EventLocation = p_location.get_text()
            EventLocation = EventLocation.strip()
            EventLocation = re.sub(r'\s+', ' ', EventLocation)

            master_list.append([EventDate, EventTitle, EventUrl, ConferenceType, EventLocation])
        except:
            pass

    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_past_conferences.csv', index=False)

    return 0


def build_master_press_release():
    # 1996 to 2005 [1996, 2006)

    master_list = []
    master_cols = ['Date', 'Title', 'Url', 'Type']

    for year in range(1996, 2006):
        year_page_url = 'https://www.federalreserve.gov/newsevents/press/all/' + str(year) + 'all.htm'
        print(year_page_url)
        response = url.urlopen(year_page_url)
        sleep(3)
        soup = bs(response, 'html.parser')
        response.close()

        ul_el = soup.find("ul", {"id": "releaseIndex"})
        ul_list = ul_el.findAll('li')

        for current_li in ul_list:
            EventDate = current_li.find(text=True, recursive=False)
            EventDate = EventDate.strip()
            EventDate = re.sub(r'\s+', ' ', EventDate)

            title_div = current_li.find("div", {"class": "indent"})
            EventTitle = title_div.get_text()
            EventTitle = EventTitle.strip()
            EventTitle = re.sub(r'\s+', ' ', EventTitle)

            EventUrl = title_div.a['href']

            EventType = "NA"

            master_list.append([EventDate, EventTitle, EventUrl, EventType])

    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_press_release_1.csv', index=False)

    page_url = 'https://www.federalreserve.gov/newsevents/pressreleases.htm'
    page_count = 191

    browser = webdriver.Chrome()
    browser.get(page_url)
    sleep(10)

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

            p_title = event_div.find("span", {"class": "itemTitle"})
            EventTitle = p_title.get_text()
            EventTitle = EventTitle.strip()
            EventTitle = re.sub(r'\s+', ' ', EventTitle)

            EventUrl = p_title.a['href']

            p_speaker = event_div.find("p", {"class": "eventlist__press"})
            EventSpeaker = p_speaker.get_text()
            EventSpeaker = EventSpeaker.strip()
            EventSpeaker = re.sub(r'\s+', ' ', EventSpeaker)
            EventType = EventSpeaker

            master_list.append([EventDate, EventTitle, EventUrl, EventType])

        master_df = pd.DataFrame(master_list, columns=master_cols)
        master_df.to_csv('master_press_release.csv', index=False)

        element = browser.find_element_by_link_text('Next')
        element.click()
        sleep(10)
    browser.close()

    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_press_release.csv', index=False)

    return 0


def build_master_current_meeting():
    index = 0
    master_list = []
    master_cols = ['Year', 'Date', 'ItemNameCommon', 'ItemName', 'Url']
    start_year = 2016
    end_year = 2022

    year_page_url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

    response = url.urlopen(year_page_url)
    soup = bs(response, 'html.parser')
    response.close()

    divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
    year = 2020
    for current_div in divs_for_cycles[1:]:
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
            print(date_text)

            minutes_divs = sub_div.find_all("div", {"class": "fomc-meeting__minutes"})
            if (len(minutes_divs) > 0):
                minutes_div = minutes_divs[0]
                links = minutes_div.findAll('a')
                if (len(links) > 0):
                    ItemNameCommon = minutes_div.strong.get_text()

                    for link in links:
                        if ItemNameCommon == None:
                            ItemNameCommon = link.get_text()
                        if ItemNameCommon.isspace():
                            ItemNameCommon = link.get_text()
                        # print(ItemNameCommon, link.get_text())
                        master_list.append([year, date_text, ItemNameCommon, link.get_text(), link['href']])

        year = year - 1
        index = index + 1
    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_FOMC_meeting_files_2016_2020_only_minutes.csv', index=False)


def download_speechs():
    start = 0
    end = 5
    master_df = pd.read_csv('master_speech_1996_2020.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('LocalPath')
    res_list = []
    err_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name_short = row['Url'].split('/')

            file_name = 'Z:/FOMC/data speeches/' + file_name_short[-1]
            if (file_name_short[-1] == 'default.htm'):
                file_name = 'Z:/FOMC/data speeches/' + 'Speech_' + file_name_short[-2] + '.htm'

            '''
            if (file_name_short[1] == 'boarddocs' and row['ItemName'] == 'Statement'):
                file_name = 'Z:/FOMC/data/' + str(row['Year']) + 'Statement_' + file_name_short[-2] + '.htm'
            '''
            print(file_name)
            if (not os.path.isfile(file_name)):
                post_fix = row['Url']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                '''
                if (post_fix == 'http://www.minneapolisfed.org/bb/'):
                    continue
                if ('http' in post_fix):
                    curr_url = post_fix
                '''
                url.urlretrieve(curr_url, file_name)
                sleep(3)

            curr_row = list(row)
            curr_row.append(file_name)
            res_list.append(curr_row)

        except Exception as e:
            print(e)
            err_list.append(list(row))
    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('master_speech.csv', index=False)
    err_df = pd.DataFrame(err_list, columns=column_list[:-1])
    err_df.to_csv('current_error.csv', index=False)


def download_testimonies():
    start = 0
    end = 5
    master_df = pd.read_csv('master_testimony_1996_2020.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('LocalPath')
    res_list = []
    err_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name_short = row['Url'].split('/')

            file_name = 'Z:/FOMC/data testimonies/' + file_name_short[-1]
            if (file_name_short[-1] == 'default.htm' or file_name_short[-1] == 'testimony.htm'):
                file_name = 'Z:/FOMC/data testimonies/' + 'Testimony_' + file_name_short[-3] + '_' + file_name_short[
                    -2] + '.htm'

            print(file_name)
            if (not os.path.isfile(file_name)):
                post_fix = row['Url']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                '''
                if (post_fix == 'http://www.minneapolisfed.org/bb/'):
                    continue
                if ('http' in post_fix):
                    curr_url = post_fix
                '''
                url.urlretrieve(curr_url, file_name)
                sleep(3)

            curr_row = list(row)
            curr_row.append(file_name)
            res_list.append(curr_row)

        except Exception as e:
            print(e)
            err_list.append(list(row))
    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('master_testimony_1996_2020.csv', index=False)
    err_df = pd.DataFrame(err_list, columns=column_list[:-1])
    err_df.to_csv('current_error.csv', index=False)


def apply_check_minutes(full_string):
    if 'Minutes' in full_string:
        return True
    else:
        return False


def apply_check_statement(full_string):
    if 'Statement' in full_string:
        return True
    else:
        return False


def apply_check_transcript(full_string):
    if 'Transcript' in full_string:
        return True
    else:
        return False


def apply_check_PC(full_string):
    if 'Press Conference' in full_string:
        return True
    else:
        return False


def apply_get_transcript_URL(URL_PC):
    full_URL_PC = 'https://www.federalreserve.gov' + URL_PC

    response = url.urlopen(full_URL_PC)
    # sleep(3)
    soup = bs(response, 'html.parser')
    response.close()

    a_tags = soup.findAll("a")
    for a_tag in a_tags:
        text = a_tag.get_text()
        if 'Press Conference Transcript' in text:
            return a_tag['href']
    return 'NF'


def build_master_current_meeting_statements():
    index = 0
    master_list = []
    master_cols = ['Year', 'Date', 'ItemNameCommon', 'ItemName', 'Url']
    start_year = 2016
    end_year = 2021

    year_page_url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

    response = url.urlopen(year_page_url)
    soup = bs(response, 'html.parser')
    response.close()

    divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
    year = 2020
    for current_div in divs_for_cycles[1:]:
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
            print(date_text)

            all_divs = sub_div.find_all("div")
            for temp_div in all_divs:
                # minutes_div = minutes_divs[0]
                temp_div_text = temp_div.get_text()
                if ('Statement:' in temp_div_text):
                    links = temp_div.findAll('a')
                    if (len(links) > 0):
                        ItemNameCommon = temp_div.strong.get_text()

                        for link in links:
                            if ItemNameCommon == None:
                                ItemNameCommon = link.get_text()
                            if ItemNameCommon.isspace():
                                ItemNameCommon = link.get_text()
                            # print(ItemNameCommon, link.get_text())
                            ItemName = link.get_text()
                            if ('Implementation Note' not in ItemName):
                                master_list.append([year, date_text, ItemNameCommon, link.get_text(), link['href']])

        year = year - 1
        index = index + 1
    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df.to_csv('master_FOMC_meeting_files_2016_2020_only_statements.csv', index=False)


def separate_meeting_master():
    ## 1996-2015
    df = pd.read_csv('master_FOMC_meeting_files_1937_2015.csv')
    df = df.loc[df['Year'] >= 1996]

    '''
    ## minutes
    df_minutes = df.loc[(df['ItemNameCommon'].apply(lambda x: apply_check_minutes(x))) | (df['ItemName'].apply(lambda x: apply_check_minutes(x)))]
    df_minutes.to_csv('master_meeting_minutes_1996_2015.csv', index=False)
    '''

    '''
    ## statements
    df_statement = df.loc[(df['ItemNameCommon'].apply(lambda x: apply_check_statement(x)))]
    df_statement.to_csv('master_meeting_statement_1996_2015.csv', index=False)
    '''

    '''
    ## transcript
    df_statement = df.loc[(df['ItemName'].apply(lambda x: apply_check_transcript(x)))]
    df_statement.to_csv('master_meeting_transcript_1996_2015.csv', index=False)
    '''

    '''
    ## Press Conferences
    df_statement = df.loc[(df['ItemNameCommon'].apply(lambda x: apply_check_PC(x)))]
    df_statement.to_csv('master_meeting_press_conference_1996_2015.csv', index=False)
    '''

    '''
    ## find press conference transcript link 
    df_PC = pd.read_csv('master_meeting_press_conference_1996_2015.csv')
    df_PC['TranscriptUrl'] = df_PC['Url'].apply(lambda x: apply_get_transcript_URL(x))
    df_PC.to_csv('master_meeting_press_conference_transcripts_1996_2015.csv', index=False)
    '''


def build_master_current_meeting_PC():
    index = 0
    master_list = []
    master_cols = ['Year', 'Date', 'ItemNameCommon', 'ItemName', 'Url']
    start_year = 2016
    end_year = 2021

    year_page_url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

    response = url.urlopen(year_page_url)
    soup = bs(response, 'html.parser')
    response.close()

    divs_for_cycles = soup.findAll("div", {"class": "panel panel-default"})
    year = 2020
    for current_div in divs_for_cycles[1:]:
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
            print(date_text)

            all_divs = sub_div.find_all("div")
            for temp_div in all_divs:
                links = temp_div.findAll('a')
                if (len(links) > 0):
                    ItemNameCommon = None

                    for link in links:
                        ItemName = link.get_text()
                        if ItemNameCommon == None:
                            ItemNameCommon = link.get_text()
                        if ItemNameCommon.isspace():
                            ItemNameCommon = link.get_text()
                        # print(ItemNameCommon, link.get_text())
                        if ('Press Conference' in ItemName):
                            master_list.append([year, date_text, ItemNameCommon, link.get_text(), link['href']])

        year = year - 1
        index = index + 1
    master_df = pd.DataFrame(master_list, columns=master_cols)
    master_df['TranscriptUrl'] = master_df['Url'].apply(lambda x: apply_get_transcript_URL(x))
    master_df.to_csv('master_FOMC_meeting_files_2016_2020_only_PC.csv', index=False)


def download_meeting_press_conference():
    start = 0
    end = 5
    master_df = pd.read_csv('master_meeting_press_conference_transcripts_2011_2020.csv')  # .iloc[start:end,:]
    column_list = list(master_df.columns)
    column_list.append('LocalPath')
    res_list = []
    err_list = []
    for index, row in master_df.iterrows():
        if not index % 10:
            print(index)
        try:
            file_name_short = row['TranscriptUrl'].split('/')

            file_name = 'Z:/FOMC/data meeting/press conference transcript/' + file_name_short[-1]

            print(file_name)
            if (not os.path.isfile(file_name)):
                post_fix = row['TranscriptUrl']
                curr_url = 'https://www.federalreserve.gov' + post_fix
                url.urlretrieve(curr_url, file_name)
                sleep(3)

            curr_row = list(row)
            curr_row.append(file_name)
            res_list.append(curr_row)

        except Exception as e:
            print(e)
            err_list.append(list(row))
    result_df = pd.DataFrame(res_list, columns=column_list)
    result_df.to_csv('master_meeting_press_conference_transcripts_2011_2020.csv', index=False)
    err_df = pd.DataFrame(err_list, columns=column_list[:-1])
    err_df.to_csv('current_error.csv', index=False)


if __name__ == '__main__':
    start = time()
    # build_master(start_year=1937, end_year=2016)
    # download_files()
    # clean_date()
    # build_master_newsevents(category = 'speech')
    # build_master_newsevents(category = 'testimony')
    # build_master_conferences()
    # build_master_press_release()
    # build_master_current_meeting()
    # download_speechs()
    # download_testimonies()
    # separate_meeting_master()
    # build_master_current_meeting_statements()
    # build_master_current_meeting_PC()
    '''
    df_temp = pd.read_csv('master_speech.csv')
    df_temp = df_temp[df_temp.duplicated(subset=['LocalPath'], keep=False)]
    print(df_temp.head())
    '''
    download_meeting_press_conference()

    print((time() - start) / 60.0)