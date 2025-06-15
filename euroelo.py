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

    default_teams = ["Liverpool", "Barcelona", "Paris SG", "Bayern Munich", "Napoli", "Sporting Lisbon"]

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

# --- TAB 2 Current ranking ---
with tab2:
    latest_date = team_elo['Date'].max()
    formatted_date = latest_date.strftime("%Y-%m-%d")
    st.markdown(f"Last update: {formatted_date}")

    # Get latest Elo per team
    latest_before_date = (
        team_elo[team_elo['Date'] <= latest_date]
        .sort_values(['Team', 'Date'])
        .groupby('Team')
        .tail(1)
    )
    latest_before_date['DaysSinceLastGame'] = (latest_date - latest_before_date['Date']).dt.days

    # Get Elo rating from ~12 months ago
    cutoff_date = latest_date - pd.DateOffset(years=1)
    ratings_12mo = []

    for team in latest_before_date['Team']:
        past_data = team_elo[(team_elo['Team'] == team) & (team_elo['Date'] <= cutoff_date)]
        if not past_data.empty:
            past_rating = past_data.sort_values('Date', ascending=False).iloc[0]['EloRating']
        else:
            past_rating = float('nan')
        ratings_12mo.append({'Team': team, 'EloRating_12mo': past_rating})

    ratings_12mo_df = pd.DataFrame(ratings_12mo)

    # Merge into latest data
    ranking = latest_before_date.merge(ratings_12mo_df, on='Team', how='left')

    # Calculate delta
    ranking['vs previous season'] = ranking['EloRating'] - ranking['EloRating_12mo']
    ranking['vs previous season'] = ranking['vs previous season'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")

    # Round EloRating and format as string with 2 decimals
    ranking['EloRating'] = ranking['EloRating'].apply(lambda x: f"{x:,.2f}")

    # Prepare and display top 50
    ranking = ranking.sort_values(by='EloRating', ascending=False).head(50)
    ranking = ranking[['Team', 'DaysSinceLastGame', 'EloRating', 'vs previous season']].reset_index(drop=True)
    ranking.insert(0, 'Rank', range(1, len(ranking) + 1))

    st.table(ranking)

# --- TAB 3 Single team last 20 games ---
with tab3:
    selected_teams = st.multiselect(
        "Select team(s) to view last 20 games",
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

        # Count games for the selected team
        team_game_count = len(team_elo[team_elo['Team'] == team])

        current_row = latest_ratings[latest_ratings['Team'] == team].iloc[0]

        # --- Rank delta ---
        if not pd.isna(current_row['Rank_12mo']):
            rank_delta_value = current_row['Rank_12mo'] - current_row['Rank']
            rank_delta_display = f"{round(rank_delta_value):+} vs previous season"
        else:
            rank_delta_value = None
            rank_delta_display = "N/A"

        # --- Rating delta ---
        if not pd.isna(current_row['EloRating_12mo']):
            rating_delta = current_row['EloRating'] - current_row['EloRating_12mo']
            rating_delta_display = f"{round(rating_delta, 2):+} vs previous season"
        else:
            rating_delta = None
            rating_delta_display = "N/A"

        # Create 3 columns
        col1, col2, col3 = st.columns(3)

        # Rank and rank delta
        col1.metric(
            label="Rank",
            value=f"#{current_row['Rank']}",
            delta=rank_delta_display if rank_delta_value is not None else None,
            delta_color="normal"  # green if rank improved (i.e., number got smaller)
        )

        # Rating and rating delta
        col2.metric(
            label="EuroElo Rating",
            value=f"{current_row['EloRating']:,.2f}",
            delta=rating_delta_display if rating_delta is not None else None
        )

        # Games in database
        col3.metric(
            label="Games in database",
            value=f"{team_game_count:,d}",
            delta=""
        )

        team_data = team_elo[team_elo['Team'] == team].sort_values(by='Date', ascending=False).head(20)
        display_data = team_data[['Date', 'Opponent', 'Competition', 'HomeAway', 'Result', 'Delta', 'EloRating']].copy()

        # Round and format 'Delta' and 'EloRating' columns
        display_data['Delta'] = display_data['Delta'].apply(lambda x: f"{x:+,.2f}" if pd.notna(x) else "N/A")
        display_data['EloRating'] = display_data['EloRating'].apply(lambda x: f"{x:,.2f}")

        st.table(display_data.reset_index(drop=True))


# --- TAB 4 Narrative presets ---
with tab4:
    st.header("Narratives (⚠️ EXPERIMENTAL ⚠️)")

    narratives = {
        "Xabi Alonso's undefeated run with Leverkusen": {
            "teams": ["Leverkusen", "Bayern Munich"],
            "start_date": datetime.date(2004, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2023, 6, 1), datetime.date(2024, 6, 1)),
            "description": "Leverkusen went undefeated and won their first-ever Bundesliga title in 2023–24, under Xabi Alonso. They fell 4 points short of closing a 224-point gap with Bayern Munich in a single season.",
        },         
        "Hansi Flick's 2020 Bayern is the GOAT": {
            "teams": ["Bayern Munich"],
            "start_date": datetime.date(2013, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2019, 11, 3), datetime.date(2021, 6, 30)),
            "description": "Bayern's 2020 team under Hansi Flick (2019-11-03 to 2021-06-30) dominated every opponent, winning the Champions League with a perfect record and crushing top teams, leading to a record high EuroElo of 2,026 on Nov 7th 2020.",
        "markers": [
                ("Bayern Munich", datetime.date(2020, 11, 7)),
            ],
        },  
        "PSG rules the Premier league (season 2024-25)": {
            "teams": ["Paris SG", "Liverpool", "Manchester City", "Aston Villa", "Arsenal"],
            "start_date": datetime.date(2024, 8, 1),
            "end_date": datetime.date.today(),
            "description": "Paris SG defeated Liverpool (1st in PL), Arsenal (2nd), Manchester City (3rd) and Aston Villa (6th) on their way to winning their first Champions League, and reached the top of the EuroElo ranking for the first time ever.",  
        }, 
        "Chelsea won the 2021 CL, but...": {
            "teams": ["Chelsea", "Manchester City"],
            "start_date": datetime.date(2020, 8, 1),
            "end_date": datetime.date(2021, 7, 31),
            "highlight_period": (datetime.date(2021, 5, 23), datetime.date(2021, 5, 30)),
            "description": "Despite winning the Champions League final against Manchester City, Chelsea ended the season below their rivals in terms of EuroElo rating.",
        },
        "French League 1 since 2000": {
            "teams": ["Lyon", "Paris SG", "Marseille", "Lille"],
            "start_date": datetime.date(2000, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2011, 8, 1), datetime.date.today()),
            "description": "Lyon heavily dominated the French league until 2007-08 (the last of a series of 7 consecutive titles). 2008 to 2011 saw four different winners in four years. Then Paris SG took over in 2012, winning 11 of the next 13 seasons.",
        },
        "Mikel Arteta might be cooking": {
            "teams": ["Arsenal", "Manchester City"],
            "start_date": datetime.date(2016, 8, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2019, 12, 20), datetime.date.today()),
            "description": "Guardiola turned an already great team into one of the best club sides ever. Arteta, on the other hand, has taken Arsenal from a mid-table slump back to the elite, and has done so more quickly in terms of Elo growth. While Pep’s success is proven with silverware, Arteta’s numbers suggest a trajectory that could lead there soon.",
        },
        "A story of two Manchesters": {
            "teams": ["Manchester City", "Manchester United"],
            "start_date": datetime.date(1999, 8, 1),
            "end_date": datetime.date(2024, 6, 1),
            "highlight_period": (datetime.date(1999, 8, 1), datetime.date(2013, 8, 1)),
            "description": "Sir Alex Ferguson's last season as United manager (2012-13) was the moment when City took over the Premier League.",
        },    
        "Treble winners": {
            "teams": ["Barcelona", "Inter Milan", "Bayern Munich", "Manchester City", "Paris SG"],
            "start_date": datetime.date(2008, 8, 1),
            "end_date": datetime.date.today(),
            "description": "With the exception of 2009-10 Inter, every treble winning team (domestic league, domestic cup and Champions League) ended the season at the top of the EuroElo ranking.",
        "markers": [
                ("Barcelona", datetime.date(2009, 5, 27)),
                ("Inter Milan", datetime.date(2010, 5, 22)),
                ("Bayern Munich", datetime.date(2013, 5, 25)),
                ("Barcelona", datetime.date(2015, 6, 6)),
                ("Bayern Munich", datetime.date(2020, 8, 23)),
                ("Manchester City", datetime.date(2023, 6, 10)),
                ("Paris SG", datetime.date(2025, 5, 31)),
            ],
        },
        "The 1950+ elite tier": {
            "teams": ["Barcelona", "Real Madrid", "Liverpool", "Bayern Munich", "Manchester City"],
            "start_date": datetime.date(2014, 8, 1),
            "end_date": datetime.date.today(),
            "description": "Only 5 clubs have crossed the 1950 EuroElo rating in the past 10 years. This can be seen as the marker of truly elite teams.",
        "markers": [
                ("Barcelona", datetime.date(2012, 4, 14)),
                ("Barcelona", datetime.date(2016, 3, 16)),
                ("Barcelona", datetime.date(2019, 5, 1)),
                ("Real Madrid", datetime.date(2014, 12, 12)),
                ("Real Madrid", datetime.date(2017, 6, 3)),
                ("Real Madrid", datetime.date(2024, 6, 1)),
                ("Bayern Munich", datetime.date(2020, 11, 7)),
                ("Liverpool", datetime.date(2020, 2, 15)),
                ("Liverpool", datetime.date(2022, 5, 3)),
                ("Manchester City", datetime.date(2021, 5, 5)),
                ("Manchester City", datetime.date(2024, 9, 14)),
            ],
        },
        "The Premier League's 'Big 6' era is over": {
            "teams": ["Arsenal", "Liverpool", "Manchester City", "Manchester United", "Tottenham", "Chelsea"],
            "start_date": datetime.date(2015, 8, 1),
            "end_date": datetime.date.today(),
            #"highlight_period": (datetime.date(2019, 12, 20), datetime.date.today()),
            "description": "Until the end of the 2016-17 season, the Big 6 was clearly above the rest of the PL. At the end of the 2024-25 season, it looks like a Big 3 more than a Big 6 (statistically: the EuroElo standard deviation of this Big 6 went from 31 in 2017 to 87 in 2025).",
        },
        "Pep, Jürgen & Carlo": {
            "teams": ["Manchester City", "Liverpool", "Real Madrid"],
            "start_date": datetime.date(2020, 7, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2021, 7, 1), datetime.date(2024, 6, 1)),
            "description": "Between 2021 and 2024, football witnessed a unique tactical triangle: Klopp’s high-intensity pressing, Guardiola’s positional control, and Ancelotti’s adaptive mastery. For three years, these three giants shaped Europe’s biggest games and trophies.",
        }, 
        "Arab Money 🛢️": {
            "teams": ["Manchester City", "Paris SG"],
            "start_date": datetime.date(1990, 1, 1),
            "end_date": datetime.date.today(),
            "highlight_period": (datetime.date(2008, 5, 23), datetime.date.today()),
            "description": "Manchester City's investment journey began in September 2008 when Sheikh Mansour bin Zayed Al Nahyan acquired the club through the Abu Dhabi United Group. In France, backed by the Qatari government, QSI acquired a majority stake in 2011 and then became the Parisian outfit's sole owner in 2012.",
            "markers": [
                ("Manchester City", datetime.date(2008, 9, 1)),
                ("Paris SG", datetime.date(2011, 6, 1)),
                ("Paris SG", datetime.date(2012, 3, 1)),
            ],
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
## Modern football culture moves fast — too fast.

One week a team is “finished.” The next, they’re “back.” On social media, every goal sparks a hot take, every clip is judged in isolation, and every post is designed to go viral — not to inform. The bigger picture gets lost.

EuroElo is a push against that trend. It’s a dynamic system that tracks the evolution of European football clubs over time. Unlike weekly power rankings or short-lived form tables, EuroElo blends both domestic and European competitions, weighing every result — from routine league matches to the Champions League final — into a consistent rating scale.

The goal isn’t to go viral. The goal is to gain perspective and understand where a team has been, how far they’ve come, and where they might be headed in the long run.

### How it works

EuroElo is a dynamic Elo-based rating system that tracks the strength of European football clubs over time. Each match affects a team's rating based on the result and the opponent's quality. Some matches carry more weight — especially later-stage games in major competitions like the Champions League. To account for this, EuroElo adjusts match weight based on:
- the competition (e.g., domestic league vs. Champions League)
- the stage of the competition (e.g., group stage vs. semi final vs. final)

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
