import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("EuroElo")
st.badge("BETA", icon=":material/experiment:", color="grey")

# --- Load data ---
team_elo = pd.read_csv('data/elo.csv', parse_dates=['Date'])
team_elo['Date'] = pd.to_datetime(team_elo['Date'], errors='coerce')
team_elo['Team'] = team_elo['Team'].astype(str)

# --- Helper function: generate season shapes and annotations ---
def generate_season_lines(start_date, end_date, y_min, y_max):
    season_lines = [pd.Timestamp(year=year, month=6, day=10) for year in range(start_date.year, end_date.year + 1)]
    shapes = []
    annotations = []
    y_text = y_max + 10

    for i, d in enumerate(season_lines):
        shapes.append(
            dict(
                type="line",
                x0=d,
                x1=d,
                y0=y_min,
                y1=y_max,
                line=dict(color="LightGray", width=1, dash="dot"),
                xref='x',
                yref='y'
            )
        )
        if i < len(season_lines) - 1:
            season_label = f"{d.year}/{str(d.year + 1)[-2:]}"
            annotations.append(
                dict(
                    x=d,
                    y=y_text,
                    xref='x',
                    yref='y',
                    text=season_label,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor='left'
                )
            )
    return shapes, annotations

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chart", "Ranking", "Team", "Narratives", "About EuroElo"])

# --- TAB 1 Elo chart with date picker ---
with tab1:
    #st.header("Elo Rating Over Time")

    min_date = team_elo['Date'].min()
    max_date = team_elo['Date'].max()
    default_start = datetime.date(2024, 8, 15)
    start_date, end_date = st.date_input(
        "Select date range for chart",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered_dates = team_elo[(team_elo['Date'] >= pd.to_datetime(start_date)) & 
                              (team_elo['Date'] <= pd.to_datetime(end_date))]

    default_teams = ["Liverpool", "Barcelona", "Paris SG", "Bayern Munich", "Napoli", "Sp Lisbon"]

    teams = st.multiselect(
        "Select teams to display",
        options=sorted(filtered_dates["Team"].unique()),
        default=default_teams
    )

    filtered_data = filtered_dates[filtered_dates['Team'].isin(teams)]

    fig = px.line(filtered_data, x='Date', y='EloRating', color='Team',
                  title=f'Elo Rating Over Time ({start_date} to {end_date})',
                  labels={'Date': 'Date', 'EloRating': 'Elo Rating'},
                  markers=True)

    y_min = filtered_data['EloRating'].min() if not filtered_data.empty else 0
    y_max = filtered_data['EloRating'].max() if not filtered_data.empty else 1000
    shapes, annotations = generate_season_lines(pd.to_datetime(start_date), pd.to_datetime(end_date), y_min, y_max)
    fig.update_layout(shapes=shapes, annotations=annotations, height=700)

    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2 Current top 20 table ---
with tab2:
    latest_date = team_elo['Date'].max()
    formatted_date = latest_date.strftime("%Y-%m-%d")

    #st.header(f"Current Top 20")
    st.markdown(f"Last update: {formatted_date}")
    latest_before_date = (
        team_elo[team_elo['Date'] <= latest_date]
        .sort_values(['Team', 'Date'])
        .groupby('Team')
        .tail(1)
    )
    latest_before_date['DaysSinceLastGame'] = (latest_date - latest_before_date['Date']).dt.days
    top_20 = latest_before_date.sort_values(by='EloRating', ascending=False).head(20)
    top_20 = top_20[['Team', 'DaysSinceLastGame', 'EloRating']].reset_index(drop=True)
    top_20.insert(0, 'Rank', range(1, len(top_20) + 1))  # Add Rank column
    st.table(top_20)  # This will now show only the columns, no index


# --- TAB 3 Single team last 10 games ---
with tab3:
    #st.header("Single team")

    selected_teams = st.multiselect(
        "Select team(s) to view last 10 games",
        options=sorted(team_elo['Team'].unique())
    )

    # Define cutoff date (12 months ago)
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=1)

    # Latest ratings (same as before)
    latest_ratings = (
        team_elo.sort_values('Date')
                .groupby('Team')
                .last()['EloRating']
                .sort_values(ascending=False)
                .reset_index()
    )
    latest_ratings['Rank'] = range(1, len(latest_ratings) + 1)

    # Ratings from ~12 months ago
    ratings_12mo = []

    for team in latest_ratings['Team']:
        # Filter team data up to cutoff date
        past_data = team_elo[(team_elo['Team'] == team) & (team_elo['Date'] <= cutoff_date)]
        if not past_data.empty:
            # Get closest date to cutoff_date (latest before cutoff)
            rating = past_data.sort_values('Date', ascending=False).iloc[0]['EloRating']
        else:
            # No data before cutoff: could assign NaN or a default
            rating = float('nan')
        ratings_12mo.append({'Team': team, 'EloRating_12mo': rating})

    ratings_12mo_df = pd.DataFrame(ratings_12mo)

    # Rank teams by 12-months-ago rating, ignoring NaNs
    ratings_12mo_df = ratings_12mo_df.dropna(subset=['EloRating_12mo'])
    ratings_12mo_df = ratings_12mo_df.sort_values('EloRating_12mo', ascending=False).reset_index(drop=True)
    ratings_12mo_df['Rank_12mo'] = range(1, len(ratings_12mo_df) + 1)

    # Merge 12mo ratings and rank into latest_ratings
    latest_ratings = latest_ratings.merge(ratings_12mo_df[['Team', 'EloRating_12mo', 'Rank_12mo']], on='Team', how='left')

    # Then, in your loop, for each team:
    for team in selected_teams:
        st.markdown(f"### {team}")

        current_row = latest_ratings[latest_ratings['Team'] == team].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Rank", f"#{current_row['Rank']}")
        col2.metric("Current Elo Rating", f"{current_row['EloRating']:.0f}")
        col3.metric("Rank 12 Months Ago", f"#{int(current_row['Rank_12mo'])}" if not pd.isna(current_row['Rank_12mo']) else "N/A")
        col4.metric("Rating 12 Months Ago", f"{int(current_row['EloRating_12mo'])}" if not pd.isna(current_row['EloRating_12mo']) else "N/A")

        team_data = team_elo[team_elo['Team'] == team].sort_values(by='Date', ascending=False).head(10)
        display_data = team_data[['Date', 'Opponent', 'HomeAway', 'Result', 'Delta', 'EloRating']]
        st.dataframe(display_data.reset_index(drop=True))


# --- TAB 4 Narrative presets ---
with tab4:
    st.header("Narratives (⚠️ EXPERIMENTAL ⚠️)")

    narratives = {
        "Xabi Alonso's historical run with Leverkusen ⬛️🟥": {
            "teams": ["Leverkusen", "Bayern Munich"],
            "start_date": datetime.date(2004, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2023, 6, 1), datetime.date(2024, 6, 1)),
            "description": "Leverkusen went undefeated and won their first-ever Bundesliga title in 2023–24, under Xabi Alonso. This season marks a major historical breakthrough.",
        },    
        "Pep Guardiola's rise to the top 🔝": {
            "teams": ["Man City", "Liverpool"],
            "start_date": datetime.date(2008, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2016, 2, 1), datetime.date(2023, 6, 1)),
            "description": "On 1 February 2016, Manchester City signed Guardiola for the start of the 2016–17 season. Guardiola guided City to their first Champions League final in 2020–21, and their first Champions League title as part of his second continental treble in 2022–23.",
            "markers": [
                ("Man City", datetime.date(2023, 6, 10)),
            ],
        },      
        "Hansi Flick's 2020 Bayern is the GOAT 🐐": {
            "teams": ["Bayern Munich"],
            "start_date": datetime.date(2013, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2019, 11, 3), datetime.date(2021, 6, 30)),
            "description": "Bayern's 2020 team under Hansi Flick (2019-11-03 to 2021-06-30) dominated every opponent, winning the Champions League with a perfect record and crushing top teams (like Barcelona 8–2), leading to a record high EuroElo of 2,002.144 on Nov 7th 2020.",
        "markers": [
                ("Bayern Munich", datetime.date(2020, 11, 7)),
            ],
        },  
        "PSG vs Premier league (season 2024-25) 🇫🇷🆚🇬🇧": {
            "teams": ["Paris SG", "Liverpool", "Man City", "Aston Villa", "Arsenal"],
            "start_date": datetime.date(2024, 8, 1),
            "end_date": datetime.date.today(),
            "description": "Paris SG defeated Liverpool (1st in PL), Arsenal (2nd), Manchester City (3rd) and Aston Villa (6th) on their way to winning their first Champions League, and reached the top of the EuroElo ranking for the first time ever.",  
        }, 
        "Chelsea won the 2021 CL, but... 🤔": {
            "teams": ["Chelsea", "Man City"],
            "start_date": datetime.date(2020, 8, 1),
            "end_date": datetime.date(2021, 7, 31),
            "highlight_period": (datetime.date(2021, 5, 23), datetime.date(2021, 5, 30)),
            "description": "Despite winning the Champions League final against Manchester City, Chelsea ended the season below their rivals in terms of EuroElo rating.",  
        },
        "French League 1 since 2000 🇫🇷": {
            "teams": ["Lyon", "Paris SG", "Marseille", "Lille"],
            "start_date": datetime.date(2000, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2011, 8, 1), datetime.date.today()),
            "description": "Lyon heavily dominated the French league until 2007-08 (the last of a series of 7 consecutive titles). 2008 to 2011 saw four different winners in four years. Then Paris SG took over in 2012, winning 11 of the next 13 seasons.",
        },
        "Mikel Arteta might be cooking ⬜️🟥🧑‍🍳": {
            "teams": ["Arsenal"],
            "start_date": datetime.date(1996, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2019, 12, 20), datetime.date.today()),
            "description": "On 20 December 2019, Mikel Arteta was appointed head coach at his former club Arsenal. Five years later, the club might be the best it's ever been since the end of the Wenger era (1996-2018), despite not winning any trophy since 2020-21 (FA Cup).",
        },
        "A story of two Manchesters 🟦🆚🟥": {
            "teams": ["Man City", "Man United"],
            "start_date": datetime.date(1999, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(1999, 8, 1), datetime.date(2013, 8, 1)),
            "description": "Since Sir Alex Ferguson left Manchester United on May 8th 2013, it's been pretty bad.",
        },    
        "Treble winners ⭐️⭐️⭐️": {
            "teams": ["Barcelona", "Inter", "Bayern Munich", "Man City", "Paris SG"],
            "start_date": datetime.date(2008, 8, 1),
            "end_date": datetime.date.today(),
            "description": "With the exception of 2009-10 Inter, every treble winning team (domestic league, domestic cup and Champions League) ended the season at the top of the EuroElo ranking.",
        "markers": [
                ("Barcelona", datetime.date(2009, 5, 27)),
                ("Inter", datetime.date(2010, 5, 22)),
                ("Bayern Munich", datetime.date(2013, 5, 25)),
                ("Barcelona", datetime.date(2015, 6, 6)),
                ("Bayern Munich", datetime.date(2020, 8, 23)),
                ("Man City", datetime.date(2023, 6, 10)),
                ("Paris SG", datetime.date(2025, 5, 31)),
            ],
        },
        "The 1950+ elite tier 🔝": {
            "teams": ["Real Madrid", "Barcelona", "Liverpool", "Bayern Munich", "Man City"],
            "start_date": datetime.date(2014, 8, 1),
            "end_date": datetime.date.today(),
            "description": "Only 5 clubs have crossed the 1950 EuroElo rating in the past 10 years - and it happened just 8 times. This can be seen as the marker of the elite-defining and dominating teams.",
        "markers": [
                ("Real Madrid", datetime.date(2014, 12, 12)),
                ("Real Madrid", datetime.date(2017, 6, 3)),
                ("Real Madrid", datetime.date(2024, 6, 1)),
                ("Barcelona", datetime.date(2016, 3, 16)),
                ("Bayern Munich", datetime.date(2020, 11, 7)),
                ("Liverpool", datetime.date(2020, 2, 15)),
                ("Man City", datetime.date(2021, 5, 5)),
                ("Man City", datetime.date(2024, 9, 14)),
            ],
        },
        "There is no Big 6 in the PL anymore": {
            "teams": ["Arsenal", "Liverpool", "Man City", "Man United", "Tottenham", "Chelsea"],
            "start_date": datetime.date(2015, 8, 1),
            "end_date": datetime.date.today(),
            #"highlight_period": (datetime.date(2019, 12, 20), datetime.date.today()),
            "description": "Until the end of the 2016-17 season, the Big 6 was clearly above the rest of the PL. Since then, two clubs (Liverpool and Man City) started distancing themselves the rest of the pack. At the end of the 2024-25 season, it looks like a Big 3 more than a Big 6.",
        },
        "Arab Money 🛢️": {
            "teams": ["Man City", "Paris SG"],
            "start_date": datetime.date(1990, 1, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2008, 5, 23), datetime.date.today()),
            "description": "Manchester City's investment journey began in September 2008 when Sheikh Mansour bin Zayed Al Nahyan acquired the club through the Abu Dhabi United Group. In France, backed by the Qatari government, QSI acquired a majority stake in 2011 and then became the Parisian outfit's sole owner in 2012.",
            "markers": [
                ("Man City", datetime.date(2008, 9, 1)),
                ("Paris SG", datetime.date(2011, 6, 1)),
                ("Paris SG", datetime.date(2012, 3, 1)),
            ],
        },
        "Messi vs Ronaldo (La Liga years) 🇪🇸": {
            "teams": ["Barcelona", "Real Madrid"],
            "start_date": datetime.date(2009, 8, 1),
            "end_date": datetime.date(2018, 7, 31),
            "description": "Overview of the years when both players were both active in La Liga.",  
        },       
    }

    selected_narrative = st.selectbox("Choose a narrative", ["None"] + list(narratives.keys()))

    if selected_narrative != "None":
        narrative_data = narratives[selected_narrative]
        st.markdown(f"### {narrative_data['description']}")

        start = pd.to_datetime(narrative_data["start_date"])
        end = pd.to_datetime(narrative_data["end_date"])
        teams = narrative_data["teams"]

        # Ensure date column is in datetime format
        team_elo['Date'] = pd.to_datetime(team_elo['Date'])

        # Filter data by date and team
        filtered_narrative = team_elo[
            (team_elo['Date'] >= start) &
            (team_elo['Date'] <= end) &
            (team_elo['Team'].isin(teams))
        ]

        # Base chart
        fig = px.line(
            filtered_narrative, x='Date', y='EloRating', color='Team',
            title=f"Elo Ratings: {selected_narrative}",
            labels={'Date': 'Date', 'EloRating': 'Elo Rating'},
            markers=True,
            height=800
        )

        # Add season dividers and annotations
        y_min = filtered_narrative['EloRating'].min() if not filtered_narrative.empty else 0
        y_max = filtered_narrative['EloRating'].max() if not filtered_narrative.empty else 1000
        shapes, annotations = generate_season_lines(start, end, y_min, y_max)
        fig.update_layout(shapes=shapes, annotations=annotations)

        # Optional highlight range
        if "highlight_period" in narrative_data:
            hl_start, hl_end = narrative_data["highlight_period"]
            fig.add_vrect(
                x0=pd.Timestamp(hl_start),
                x1=pd.Timestamp(hl_end),
                fillcolor="LightPink",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f"Highlight: {hl_start} to {hl_end}",
                annotation_position="top left"
            )
        # ➕ Add narrative-specific markers
        if "markers" in narrative_data:
            import plotly.graph_objects as go

            for team_name, event_date in narrative_data["markers"]:
                event_date = pd.to_datetime(event_date)
                team_data = filtered_narrative[filtered_narrative["Team"] == team_name]

                if not team_data.empty:
                    closest_row = team_data.iloc[(team_data["Date"] - event_date).abs().argsort()[:1]]
                    x = pd.to_datetime(closest_row["Date"].values[0])
                    y = closest_row["EloRating"].values[0]

                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='text',
                        text=[f"{team_name} ⭐️"],
                        textposition="top left",
                        name=f"{team_name} marker",
                        showlegend=False,
                        hoverinfo="skip",
                    ))

        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Select a narrative to show filtered chart.")

# --- TAB 5 About section ---
with tab5:
    st.markdown(
        """ 
        ## About EuroElo

    TLDR: EuroElo essentially treats all European clubs as if they were competing in a single, unified competition—regardless of the actual leagues or cups they play in. By combining match results across domestic leagues, cups, and European tournaments and applying a consistent rating update method, it ranks teams on one continuous scale.

    Long version: EuroElo is a dynamic rating system that tracks the strength of European football clubs over time. Each match affects a team's rating based on the result and the opponent's quality. Some matches carry more weight — especially later-stage games in major competitions like the Champions League. To account for this, EuroElo adjusts match weight based on:
    - the competition (e.g., league vs. Champions League)
    - the stage of the competition (e.g., group stage vs. final)

    As tournaments progress, matches have a bigger impact — especially in the final rounds.

    | Stage     | Domestic league | Domestic cup | Conference League | Europa League | Champions League |
    |-----------|-----------------|--------------|-------------------|---------------|------------------|
    | 0 (group) | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜    | 🟩⬜⬜⬜⬜          | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜           |
    | 1 (1/16)  | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜    | 🟩⬜⬜⬜⬜          | 🟩⬜⬜⬜⬜       | 🟩🟩⬜⬜⬜           |
    | 2 (1/8)   | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜    | 🟩⬜⬜⬜⬜          | 🟩🟩⬜⬜⬜       | 🟩🟩⬜⬜⬜           |
    | 3 (1/4)   | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜    | 🟩⬜⬜⬜⬜          | 🟩🟩⬜⬜⬜       | 🟩🟩⬜⬜⬜           |
    | 4 (1/2)   | 🟩⬜⬜⬜⬜       | 🟩⬜⬜⬜⬜    | 🟩🟩⬜⬜⬜          | 🟩🟩⬜⬜⬜       | 🟩🟩🟩⬜⬜           |
    | 5 (final) | 🟩⬜⬜⬜⬜       | 🟩🟩⬜⬜⬜    | 🟩🟩⬜⬜⬜          | 🟩🟩🟩⬜⬜       | 🟩🟩🟩🟩⬜           |

    This model is designed to:
    - Reward consistent success
    - Recognize deep runs in European competitions
    - Reflect the true prestige of key victories, like winning a final trophy

    It balances long-term form with short-term impact, creating a fair and evolving picture of who the strongest teams in Europe really are.
        """
    )

    num_teams = team_elo['Team'].nunique()
    num_games = len(team_elo) // 2  # Each match has 2 rows (home & away)
    st.markdown(f"**📊 Dataset:** {num_teams} teams, {num_games} games")

    st.markdown("---")
    st.markdown("📬 [Feedback / questions / remarks / suggestions? Leave it here](https://forms.gle/Ad4ygEGku7dzeycL8)")
