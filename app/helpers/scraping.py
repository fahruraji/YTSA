import googleapiclient.discovery
import pandas as pd
import datetime
import re

from app import app

date_format = "%Y-%m-%d"

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEYS = app.config['DEVELOPER_KEYS']

youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEYS)


def is_url(text):
    url_pattern = re.compile(r'https?://\S+')
    return bool(re.match(url_pattern, text))

def is_youtube_url(text):
    youtube_url_pattern = re.compile(r'https?://www\.youtube\.com/watch\?v=')
    return bool(re.search(youtube_url_pattern, text))

def get_ids(query):
    vidIds =  youtube.search().list(
                    part="id",
                    type='video',
                    order='viewCount',
                    q=query,
                    maxResults=50,
                    fields="items(id(videoId))"
        ).execute()
    
    return get_infos(vidIds)

def get_info(vidId):
    videos_info = []

    r = youtube.videos().list(
        part="statistics,snippet",
        id=vidId,
        fields="items(statistics," + \
                    "snippet(title,publishedAt,description,thumbnails))"
    ).execute()
    
    try:
        thumbnail = r['items'][0]['snippet']['thumbnails']['medium']['url'] 
        title = r['items'][0]['snippet']['title']
        publishedAt = r['items'][0]['snippet']['publishedAt']
        description = r['items'][0]['snippet']['description']
        views = r['items'][0]['statistics']['viewCount']
        likes = r['items'][0]['statistics']['likeCount']
        favorites = r['items'][0]['statistics']['favoriteCount']
        comments = r['items'][0]['statistics']['commentCount']
        
        publishedAt = datetime.datetime.strptime(publishedAt,"%Y-%m-%dT%H:%M:%SZ")
        
        videos_info.append({
            'thumbnail' : thumbnail,
            'id' : vidId,
            'title' : title,
            'publishedAt' : publishedAt.strftime(date_format),
            'description' : description,
            'views' : views,
            'likes' : likes,
            'favorites' : favorites,
            'comments' : comments,
        })
        
    except:
        pass
        
    return videos_info

def get_infos(vidIds):
    videos_info = []

    for item in vidIds['items']:
        vidId = item['id']['videoId']
        r = youtube.videos().list(
            part="statistics,snippet",
            id=vidId,
            fields="items(statistics," + \
                        "snippet(title,publishedAt,description,thumbnails))"
        ).execute()
        
        try:
            thumbnail = r['items'][0]['snippet']['thumbnails']['medium']['url'] 
            title = r['items'][0]['snippet']['title']
            publishedAt = r['items'][0]['snippet']['publishedAt']
            description = r['items'][0]['snippet']['description']
            views = r['items'][0]['statistics']['viewCount']
            likes = r['items'][0]['statistics']['likeCount']
            favorites = r['items'][0]['statistics']['favoriteCount']
            comments = r['items'][0]['statistics']['commentCount']
            
            publishedAt = datetime.datetime.strptime(publishedAt,"%Y-%m-%dT%H:%M:%SZ")
            
            videos_info.append({
                'thumbnail' : thumbnail,
                'id' : vidId,
                'title' : title,
                'publishedAt' : publishedAt.strftime(date_format),
                'description' : description,
                'views' : views,
                'likes' : likes,
                'favorites' : favorites,
                'comments' : comments,
            })
            
        except:
            pass
        
    return videos_info

def scrape_comments(vidId):
    cols = []
    
    r = youtube.videos().list(
        part="snippet",
        id=vidId,
        fields="items(snippet(title))"
    ).execute()
    
    data = youtube.commentThreads().list(
        part='snippet', 
        videoId=vidId, 
        maxResults='100', 
        textFormat="plainText"
    ).execute()

    while True:
        for i in data["items"]:
            name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
            comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
            published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
            likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
            replies = i["snippet"]['totalReplyCount']
            published_at = datetime.datetime.strptime(published_at,"%Y-%m-%dT%H:%M:%SZ")
            title = r['items'][0]['snippet']['title']
            cols.append({
                'name': name,
                'comment': comment,
                'published_at': published_at.strftime(date_format),
                'likes': likes,
                'replies': replies,
                'title': title,
                'Id': vidId
            })
            
            totalReplyCount = i["snippet"]['totalReplyCount']
            if totalReplyCount > 0:
                parent = i["snippet"]['topLevelComment']["id"]
                data2 = youtube.comments().list(
                    part='snippet', 
                    maxResults='100', 
                    parentId=parent,
                    textFormat="plainText"
                ).execute()
                for i in data2["items"]:
                    name = i["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["textDisplay"]
                    published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ''
                    published_at = datetime.datetime.strptime(published_at,"%Y-%m-%dT%H:%M:%SZ")
                    title = r['items'][0]['snippet']['title']
                    cols.append({
                        'name': name,
                        'comment': comment,
                        'published_at': published_at.strftime(date_format),
                        'likes': likes,
                        'replies': replies,
                        'title': title,
                        'Id': vidId
                    })

        if "nextPageToken" not in data:
            break

        data = youtube.commentThreads().list(
            part='snippet', 
            videoId=vidId, 
            pageToken=data["nextPageToken"], 
            maxResults='100', 
            textFormat="plainText"
        ).execute()

    return cols