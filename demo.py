import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain_community.document_loaders.csv_loader import CSVLoader
from operator import itemgetter
from qdrant_client import models
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import requests
from langchain.prompts import ChatPromptTemplate
import ast
import json
import re
import folium
import pandas as pd
from math import radians, cos
import numpy as np


# Change api key
os.environ["OPENAI_API_KEY"] = "your_api_key"
# Get from the api key from https://geocode.maps.co/
geo_api_key = "your_api_key"

# Change the path from here
input_path = 'your_path'
full_path='ypur_path'

# Select a city name form one of [Philadelphia,Indianapolis,Santa Barbara,Saint Louis,Nashville]
city_selected = 'Indianapolis'



# Enable Debug mode
# import langchain
# langchain.debug = True
# langchain.verbose = True



class ChatbotWithRetrieval:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
        self.conversation_history = ""
        self.global_docs_content = "" 
        self.ans = [] 
        self.markers = []
        self.llm_dic = {}
        self.cities_info = {
            'Philadelphia': {
                'state': 'PA',
                'latitude': 39.9526,
                'longitude': -75.1652
            },
            'Indianapolis': {
                'state': 'IN',
                'latitude': 39.7684,
                'longitude': -86.1581
            },
            'Santa Barbara': {
                'state': 'CA',
                'latitude': 34.4208,
                'longitude': -119.6982
            },
            'Saint Louis': {
                'state': 'MO',
                'latitude': 38.6270,
                'longitude': -90.1994
            },
            'Nashville': {
                'state': 'TN',
                'latitude': 36.1627,
                'longitude': -86.7816
            }
        }



        file_path = input_path
        metadata_columns = ['business_id','name','longitude', 'latitude','address','categories','hours','stars','tips_summary']
        loader = CSVLoader(
            file_path=file_path,
            metadata_columns=metadata_columns,
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
            }
        )

        data = loader.load()
        for doc in data:
            doc.metadata['longitude'] = float(doc.metadata['longitude'])
            doc.metadata['latitude'] = float(doc.metadata['latitude'])

        # store data in vector database
        self.vectorstore = Qdrant.from_documents(
            documents=data, 
            embedding=self.embeddings, 
            path = '/Users/victor/Desktop/vectorDB2', 
            collection_name="yelp_colls",
            force_recreate=False) 


        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system","You are an assistant for location information sorting tasks. Below is the location information retrieved from the database, which will be given to you in JSON format. You are asked to filter and sort this information based on the question asked. You first need to determine whether the information is relevant to the question, and then sort all the relevant information. The ones that best match the question and help answer it have the highest priority. The format of your output must be a Python dictionary, where the key is the name of the location and the value is the reason why you chose this location and ranked it there. The location with the highest priority is placed higher, i.e., index is 0. Please note that there could be more than one result in the dictionary. If the information about a location could only partially match the question asked, you could also put it in the dictionary, but specify the advantages and disadvantages of this place in the value of the dictionary. If you could not complete the task or do not know the answer, just return the empty dictionary and do not refer to any additional knowledge"
                ),
                ("human", "information:{context}\nquestion:{question}"),
            ]
        )


        self.qa_chain = rag_prompt| self.llm| StrOutputParser()
 



    def get_coordinates(self, suburb, city='Saint Louis'):
        city_info = self.cities_info[city]
        state = city_info['state']

        formatted_suburb = suburb.replace(' ', '+')+ '+' + city.replace(' ', '+') + f"+{state}+US"
        url = f"https://geocode.maps.co/search?q={formatted_suburb}&api_key={geo_api_key}"

        latitude =  city_info['latitude']
        longitude = city_info['longitude']
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                latitude = float(data[0]['lat'])
                longitude = float(data[0]['lon'])
                return latitude, longitude
            else:
                return latitude, longitude
        else:
            return latitude, longitude
        
        
    def normalize_text(self,text):
        return re.sub(r'\W+', '', text.lower())

    def find_coordinates(self,rank_dict,ans):
        coordinates_dict = {}
        data_for_table = []
        normalized_rank_dict = {self.normalize_text(key): (value, key) for key, value in rank_dict.items()}
        print("ccccccccccccccccccccccccc")
        for document in ans:
            normalized_place_name = self.normalize_text(document.metadata['name'])

            longitude = document.metadata['longitude']
            latitude = document.metadata['latitude']
            original_name = document.metadata['name']
            stars= document.metadata['stars']
            categorie = document.metadata['categories']
            address= document.metadata['address']
            hours = document.metadata['hours']

            
            if normalized_place_name in normalized_rank_dict:
                rank_value, original_name = normalized_rank_dict[normalized_place_name]
                
                coordinates_dict[original_name] = {
                    'longitude': longitude,
                    'latitude': latitude,
                    'llm_elected':True,
                    'text': rank_value,
                }

                data_for_table.append({
                    "Place Name": original_name,
                    "Address":address,
                    "Categorie":categorie,
                    'Stars':stars,
                    "LLM_recommend": True,
                    "Opening hours":hours,
                    'Longitude':longitude,
                    'Latitude':latitude,
                    "Details": rank_value,
                })

        for document in ans:
            normalized_place_name = self.normalize_text(document.metadata['name'])
            longitude = document.metadata['longitude']
            latitude = document.metadata['latitude']
            original_name = document.metadata['name']
            stars= document.metadata['stars']
            categorie = document.metadata['categories']
            address= document.metadata['address']
            hours = document.metadata['hours']
            if normalized_place_name not in normalized_rank_dict:
                text = "This is what the vector search looks up, our system AI doesn't recommend this place, but you can check it out as well."
                coordinates_dict[original_name] = {
                    'longitude': longitude,
                    'latitude': latitude ,
                    'text': document.metadata['tips_summary'],
                    'llm_elected': False
                }

                data_for_table.append({
                    "Place Name": original_name,
                    "Address":address,
                    "Categorie":categorie,
                    'Stars':stars,
                    "LLM_recommend": False,
                    "Opening hours":hours,
                    'Longitude':longitude,
                    'Latitude':latitude,
                    "Details": document.metadata['tips_summary'],
                })

        print(data_for_table)
        return coordinates_dict,data_for_table
    
            
    def find_query(self,_dict,side_km = 5):
        query = _dict['query']
        lat = _dict['lat']
        lon = _dict['lon']

        half_side_km = side_km / 2
        delta_lat = half_side_km / 111  
        delta_lon = half_side_km / (111 * np.cos(radians(lat)))

        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.latitude",
                    range=models.Range(
                        gte=lat - delta_lat,  
                        lte=lat + delta_lat,  
                    ),
                ),
                models.FieldCondition(
                    key="metadata.longitude",
                    range=models.Range(
                        gte=lon - delta_lon,  
                        lte=lon + delta_lon,  
                    ),
                )
            ]
        )
        ans_test = self.vectorstore.similarity_search(query,k=8,filter=filter)
        return ans_test 
    
    

    def get_response(self, user_input, suburb):  
        latitude, longitude = self.get_coordinates(suburb)
        print("the coordinates is ..")
        print(latitude)
        print(longitude)
        self.m = folium.Map(location=[latitude, longitude], zoom_start=13) 
        dictest = {"query": user_input, "lat": latitude, "lon": longitude}
        ret = self.find_query(dictest)

        docs_content = []
        initial_process_text = "Initial position marked, processing with AI, please wait..."


        for doc in ret:
            docs_content.append(doc.page_content)
            longitude = doc.metadata['longitude']
            latitude = doc.metadata['latitude']
            html = f"""
            <div style="min-width:200px; border: 1px solid black; padding: 5px;">
                <strong>{doc.metadata['name']}</strong><br>
                'Initial Location'
            </div>
            """
            iframe = folium.IFrame(html=html, width=250, height=120) 
            popup = folium.Popup(iframe, max_width=265)  
            folium.Marker(
                [latitude,longitude],
                popup=popup,
                tooltip=doc.metadata['name'],
                icon=folium.Icon(color='blue')
            ).add_to(self.m)

        initial_map_html = self.m._repr_html_()
        # yield self.conversation_history,initial_map_html, None
        yield self.conversation_history, initial_process_text ,initial_map_html, None

        context = json.dumps(docs_content, ensure_ascii=False, indent=4)
        print(context)
        response = self.qa_chain.invoke({"context": context, "question": user_input})

        def extract_dict_from_codeblock(code_str):
            start_marker = '```python\n'
            end_marker = '\n```'

            if '```python' in code_str:
                start_idx = code_str.find(start_marker)
                if start_idx == -1:
                    raise ValueError("Error, no ```python")
                end_idx = code_str.find(end_marker, start_idx)
                if end_idx == -1:
                    raise ValueError("Error, no ending ``` after ```python")
                dict_str = code_str[start_idx + len(start_marker):end_idx].strip()
            else:
                dict_str = code_str.strip()


            try:
                data_dict = json.loads(dict_str)
            except json.JSONDecodeError:
                try:
                    data_dict = ast.literal_eval(dict_str)
                except Exception as e:
                    raise ValueError(f"JSON Error: {e}")

            return data_dict


        try:
            rank_dict = extract_dict_from_codeblock(response)
        except ValueError:
            rank_dict = {}


        rank_list = list(rank_dict.items())

        
        if rank_dict:
            best_key, best_value = rank_list[0]
            if(len(rank_list)==1):
                self.conversation_history += f"The place that best fits your question is {best_key}. {best_value} \n"
            else:
                self.conversation_history += f"The place that best fits your question is {best_key}. {best_value}There are a number of other places that may be of interest to you, please refer to the map.\n"
        else:
            self.conversation_history += f"Sorry, we didn't find a suitable location in our database.\n"
            final_map_html = self.m._repr_html_()
            final_process_text = "AI processing is complete! Our system doesn't have any recommended locations."
            return self.conversation_history,  final_process_text, final_map_html, None
    


        data_for_table = []
        print(rank_dict)
        coordinates, data_for_table = self.find_coordinates(rank_dict, ret)
        columns = ["Place Name",
                    "Address",
                    "Categorie",
                    'Stars',
                    "LLM_recommend",
                    "Opening hours",
                    'Longitude',
                    'Latitude',
                    "Details"] 
        results_df = pd.DataFrame(data_for_table,columns=columns)


 

        for place_name, details in coordinates.items():
            if details['llm_elected']:
                icon = folium.Icon(color='green') 
            else:
                icon = folium.Icon(color='blue')

            html = f"""
            <div style="min-width:200px; border: 1px solid black; padding: 5px;">
                <strong>{place_name}</strong><br>
                {details['text']}
            </div>
            """
            iframe = folium.IFrame(html=html, width=250, height=120) 
            popup = folium.Popup(iframe, max_width=265)  
            folium.Marker(
                [details['latitude'], details['longitude']],
                popup=popup,
                tooltip=place_name,
                icon=icon
            ).add_to(self.m)

        final_map_html = self.m._repr_html_()

        final_process_text = "AI processing is complete! Places marked in green are AI-recommended locations."
        # yield self.conversation_history, final_map_html, results_df 
        yield self.conversation_history, final_process_text, final_map_html, results_df 
    
   

    def select(self, df, data: gr.SelectData):
        selected_index = data.index[0]
        selected_row = df.iloc[selected_index]



        lat = float(selected_row['Latitude'])
        lon = float(selected_row['Longitude'])
        map = folium.Map(location=[lat,lon], zoom_start=13)
    

        for index, row in df.iterrows():
            show_popup = False
            if row['LLM_recommend']:
                icon = folium.Icon(color='green')
    
            else:
                icon = folium.Icon(color='blue')

            if index == selected_index:
                show_popup = True
                
        
            html = f"""
            <div style="min-width:200px; border: 1px solid black; padding: 5px;">
                <strong>{row['Place Name']}</strong><br>
                {row['Details']}
            </div>
            """
            iframe = folium.IFrame(html=html, width=250, height=120) 
            popup = folium.Popup(iframe, max_width=265,show=show_popup)  
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=popup,
                tooltip=row['Place Name'],
                icon=icon
            ).add_to(map)
            

        return map._repr_html_()

    

bot = ChatbotWithRetrieval()

df2 = pd.read_csv(full_path)
suburbs = unique_neighborhoods = df2['neighborhood'].dropna().unique().tolist()

with gr.Blocks() as demo:
    gr.Markdown("# SemaSK")
    with gr.Row():
        query_input = gr.Textbox(label="Query")
        suburb_input = gr.Dropdown(label="Suburb", value = None, choices=suburbs)
        with gr.Column():
            clear_button = gr.ClearButton(components=[query_input,suburb_input],value="Clear")
            submit_button = gr.Button(value="Submit",variant="primary")
 
    with gr.Row():
        with gr.Column():
            llm_res = gr.components.Textbox(label="Response")
            process = gr.components.Textbox(label="Process")
        map_component = gr.components.HTML(label="Map")
    with gr.Row():
        data_table = gr.Dataframe(
                            headers=["LLM_recommend","Place Name","Address","Categorie",'Stars','Opening hours','Longitude','Latitude',"Details"],
                            col_count=(9, "fixed"),
                        )

    submit_button.click(
        fn=bot.get_response,
        inputs=[query_input, suburb_input],
        outputs=[llm_res, process,map_component, data_table]
    )
    data_table.select(
        fn=bot.select,
        scroll_to_output = False,
        inputs=[data_table],
        outputs=[map_component]
    )
        

if __name__ == "__main__":
    demo.launch()

