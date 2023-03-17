## Meeting Minutes Data
 - File name: aggregate_measure_mm.xlsx
 - Varibales: 
   - Year: year of the meeting
   - Date: meeting dates
   - StartDate: meeting start date
   - EndDate: meeting end date
   - ReleaseDate: the date on which meeting minutes was released
   - Url: url on FOMC website
   - labeled_data_path: path of file stored locally after labeling
   - our_measure: hawkish-dovish measure (higher value means Fed is more hawkish)

## Speeches
 - File name: aggregate_measure_sp.xlsx
 - Variables: 
   - Date: date on which speech was given
   - Title: title of the speech
   - Speaker: speaker name with title (Chair, Governor, etc)
   - Location: place at which speech was given
   - LocalPath: path to raw file in local repo
   - labeled_data_path: path of file stored locally after labeling
   - our_measure: hawkish-dovish measure (higher value means Fed is more hawkish)

 - Note: It doesn't contain all speeches, we filtered out some speeches. Details are available in the paper. If required we have all other speeches downloaded as well. 

## Press Conference: 
 - File name: aggregate_measure_pc.xlsx
 - Variables:
   - Year: year of the meeting
   - Date: meeting dates
   - StartDate: meeting start date
   - EndDate: meeting end date, also the date of press conference
   - TranscriptUrl: url from which transcript was downloaded
   - labeled_data_path: path of file stored locally after labeling
   - our_measure: hawkish-dovish measure (higher value means Fed is more hawkish)