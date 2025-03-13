import pandas as pd
import re
import os
from pybaseball import statcast
from pybaseball import playerid_reverse_lookup
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Flowable
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
import spacy
from spacy import displacy
from spacy.matcher import Matcher

def fetch_statcast_data(game_date, home_team, away_team):
    print(f"📥 Fetching play-by-play data for {game_date}...")
    df = statcast(start_dt=game_date, end_dt=game_date)

    if df.empty:
        print("⚠ No data found for this date. Please check the game date and try again.")
        return None
     # ✅ Filter data to only include rows for the specified teams
    df = df[(df['home_team'] == home_team) & (df['away_team'] == away_team)]

    if df.empty:
        print(f"⚠ No game found for {home_team} vs {away_team} on {game_date}. Check team abbreviations.")
        return None

    # ✅ Standardize inning formatting
    df = df.sort_values(by=['inning', 'at_bat_number'], ascending=[True, True]).reset_index(drop=True)
    
    return df

# Log unrecognized play descriptions for future review
UNKNOWN_PLAYS_LOG = "unknown_plays.txt"

def log_unknown_play(description):
    """ Logs unknown play descriptions for later review. """
    with open(UNKNOWN_PLAYS_LOG, "a") as file:
        file.write(description + "\n")
    print(f"LOGGED: Unrecognized Play -> {description}")

# Use spacy NLP to process the description
nlp = spacy.load('en_core_web_sm')

# Standardize inning values
def refine_inning(inn):
    if isinstance(inn, str):
        match = re.search(r'\b(top|bottom)\s+of\s+the\s+(\d+)', inn, re.IGNORECASE)
        if match:
            side = 't' if 'top' in match.group(1).lower() else 'b'
            inning_number = match.group(2)
            return f"{side}{inning_number}"
        elif inn.lower().startswith(('t', 'b')) and inn[1:].isdigit():
            return inn
    return None

def extract_player_name(description, batter_id, id_to_name):

##    Extracts the first proper noun (likely the player's name) from the description.

    # ✅ Check if the batter ID exists in the lookup dictionary
    if batter_id in id_to_name:
        return id_to_name[batter_id]  # ✅ Use the official name from MLBAM ID lookup

# Parse play description outcomes
def parse_play_description(events, description):
    if not isinstance(description, str) or description.strip() == "":
        return '-'
    
    if not isinstance(events,str) or events.strip() == "":
        return '-'
    
    # doc = nlp(description.lower()) #Process the description using spacy
    events = events.lower()
    description = description.lower()

    description = re.sub(r'\(.*?\)', '', description)
    description = re.sub(r'\b(deep|short|short|weak|thru|hole|sharply)\b', '', description)
    description = description.strip()

    print(f"DEBUG: Cleaned description -> {description}")

    event_mappings = {
        'single': '1B',
        'double': '2B',
        'triple': '3B',
        'home_run': 'HR',
        'walk': 'BB',
        'hit_by_pitch': 'HBP',
        'catcher_interf': 'CI',
        'intentional_walk': 'IBB',
    }

    # Extract Named Entities
    # entities = [ent.text.lower() for ent in doc.ents]
    # tokens = [token.text.lower() for token in doc]

    # **DEBUG PRINT: Check Description Processing**
    # print(f"DEBUG: Processing Description -> {description}")

    # **Explicitly Recognize Common Unique Plays**
    outcome = event_mappings.get(events, None)
    
    if 'hit by pitch' in description:
        return 'HBP'
    elif 'balk' in description:
        return 'BK'
    elif 'wild pitch' in description or 'wp' in description:
        return 'WP'
    elif 'passed ball' in description or 'pb' in description:
        return 'PB'
    elif 'fielder choice' in description or 'fc' in description:
        return 'FC'
    
    # **Rule-Based NLP Play Recognition**
    if 'strikeout' in events:
        if 'looking' in description or 'called' in description:
            return 'Ʞ'
        elif 'swinging' in description:
            return 'K'
        return 'K'

    # **Line Outs**
    if ('lines out' in description or 'lineout' in description) and ('cf' in description or 'center field' in description or 'center-field' in description):
        return 'L8'
    elif ('lines out' in description or 'lineout' in description) and ('rf' in description or 'right field' in description or 'right-field' in description):
        return 'L9'
    elif ('lines out' in description or 'lineout' in description) and ('lf' in description or 'left field' in description or 'left-field' in description):
        return 'L7'
    elif ('lines out' in description or 'lineout' in description) and ('ss' in description or 'shortstop' in description or 'short stop' in description):
        return 'L6'
    elif ('lines out' in description or 'lineout' in description) and ('2b' in description or 'second base' in description or 'second baseman' in description):
        return 'L4'
    elif ('lines out' in description or 'lineout' in description) and ('3b' in description or 'third base' in description or 'third baseman' in description):
        return 'L5'
    elif ('lines out' in description or 'lineout' in description) and ('1b' in description or 'first base' in description or 'first baseman' in description):
        return 'L3'
    elif ('lines out' in description or 'lineout' in description) and ('p' in description or 'pitcher' in description):
        return 'L1'
        
   # **Flyouts & Groundouts**
    if 'flies out' in description or 'flyball' in description:
        if 'left field' in description or 'lf' in description:
            return 'F7'
        elif 'center field' in description or 'cf' in description:
            return 'F8'
        elif 'right field' in description or 'rf' in description:
            return 'F9'
        
    #Sac Fly's
    if 'sac_fly'in events:
        if 'left field' in description or 'lf' in description or 'leftfield' in description:
            return 'SF7'
        if 'center field' in description or 'cf' in description or 'centerfield' in description:
            return 'SF8'
        if 'right field' in description or 'rf' in description or 'rightfield' in description:
            return 'SF9'
        
    #Double-Plays
    if 'grounded_into_double_play' in events:
        return parse_double_play(description)
    
    #Force-Outs
    if 'force_out' in events:
        return parse_force_out(description)
    
    #Fielders Choices
    if 'fielders_choice' in events:
        return parse_fielders_choice(description)
    
    #Reached-On Errors
    if 'field_error' in events:
        return parse_reached_on_error(description)

    # **Groundouts**
    if ('grounds out' in description or 'groundout' in description):
        return parse_groundouts(description)

     # **Popouts**
    if ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('catcher' in description or 'c ' in description):
        return 'P2'
    elif ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('first baseman' in description or '1b' in description):
        return 'P3'
    elif ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('second baseman' in description or '2b' in description):   
        return 'P4'
    elif ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('third baseman' in description or '3b' in description):
        return 'P5'
    elif ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('shortstop' in description or 'short stop' in description or 'ss' in description):
        return 'P6'
    elif ('pop flies' in description or 'pops out' in description or 'pop fly' in description) and ('pitcher' in description or 'p' in description):
        return 'P1'
 
    
    # **Errors**
    # if 'error' in description:
    #     for token in tokens:
    #         if token.startswith('e') and token[1:].isdigit():  # Matches "E1", "E2", etc.
    #             return token.upper()
            
    # **If no match, try extracting abbreviation (e.g., "CI", "WP")**
    # match = re.findall(r'\b[A-Z]{2,3}\b', description.upper())  # Find capitalized abbreviations
    # if match:
    #     print(f"INFO: Extracted Abbreviation -> {match[0]}")
    #     return match[0]  # ✅ Return first abbreviation found
    
     # **If still no match, log it and return '-'**
    print(f"WARNING: Unrecognized Play -> {description}")
    log_unknown_play(description)  # ✅ Log unrecognized play
    
    return outcome if outcome else '-'

# ✅ Function to Parse Double Plays
def parse_double_play(description):
    """Identifies double plays and maintains the correct order of fielders."""
    fielders = {
        'pitcher': '1', 'catcher': '2', 'first baseman': '3', 'second baseman': '4', 'third baseman': '5',
        'shortstop': '6', 'left fielder': '7', 'center fielder': '8', 'right fielder': '9'
    }

     # ✅ Extract fielder names in order from the sentence
    fielders_found = re.findall(r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)", description)

    # ✅ Convert fielders to their corresponding numbers
    involved_fielders = [fielders[f] for f in fielders_found if f in fielders]

    # ✅ Remove only **consecutive** duplicates, but allow non-consecutive repeats
    cleaned_fielders = [involved_fielders[0]]
    for i in range(1, len(involved_fielders)):
        if involved_fielders[i] != involved_fielders[i - 1]:  # Only keep non-consecutive duplicates
            cleaned_fielders.append(involved_fielders[i])

    # ✅ Ensure proper force-out format
    if len(cleaned_fielders) >= 3:
        return f"DP{'-'.join(involved_fielders[:3])}"
    elif len(cleaned_fielders) >= 2:
        return f"FO{'-'.join(cleaned_fielders)}"
    elif len(cleaned_fielders) == 1:
        return f"FO{cleaned_fielders[0]}U"  # Unassisted Force Out

    return "DP"

# ✅ Function to Parse Force Outs
def parse_force_out(description):
    """Identifies force outs with fielder positions."""
    fielders = {
        'pitcher': '1', 'catcher': '2', 'first baseman': '3', 'second baseman': '4', 'third baseman': '5',
        'shortstop': '6', 'left fielder': '7', 'center fielder': '8', 'right fielder': '9'
    }

     # ✅ Extract fielder names in order from the sentence
    fielders_found = re.findall(r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)", description)

    # ✅ Convert fielders to their corresponding numbers
    involved_fielders = [fielders[f] for f in fielders_found if f in fielders]

    # ✅ Remove only **consecutive** duplicates, but allow non-consecutive repeats
    cleaned_fielders = [involved_fielders[0]]
    for i in range(1, len(involved_fielders)):
        if involved_fielders[i] != involved_fielders[i - 1]:  # Only keep non-consecutive duplicates
            cleaned_fielders.append(involved_fielders[i])

    # ✅ Ensure proper force-out format
    if len(cleaned_fielders) >= 2:
        return f"FO{'-'.join(cleaned_fielders)}"
    elif len(cleaned_fielders) == 1:
        return f"FO{cleaned_fielders[0]}U"  # Unassisted Force Out
    
    return "FO"

# ✅ Function to Parse Fielder’s Choice
def parse_fielders_choice(description):
    """Identifies fielder's choice plays with fielder positions."""
    fielders = {
        'pitcher': '1', 'catcher': '2', 'first baseman': '3', 'second baseman': '4', 'third baseman': '5',
        'shortstop': '6', 'left fielder': '7', 'center fielder': '8', 'right fielder': '9'
    }

     # ✅ Extract fielder names in order from the sentence
    fielders_found = re.findall(r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)", description)

    # ✅ Convert fielders to their corresponding numbers
    involved_fielders = [fielders[f] for f in fielders_found if f in fielders]

    # ✅ Remove only **consecutive** duplicates, but allow non-consecutive repeats
    cleaned_fielders = [involved_fielders[0]]
    for i in range(1, len(involved_fielders)):
        if involved_fielders[i] != involved_fielders[i - 1]:  # Only keep non-consecutive duplicates
            cleaned_fielders.append(involved_fielders[i])

    # ✅ Ensure proper force-out format
    if len(cleaned_fielders) >= 2:
        return f"FC{'-'.join(cleaned_fielders)}"
    elif len(cleaned_fielders) == 1:
        return f"FC{cleaned_fielders[0]}U"  # Unassisted Force Out

    return "FC"

def parse_groundouts(description):
    fielders = {
        'pitcher': '1', 'catcher': '2', 'first baseman': '3', 'second baseman': '4', 
        'third baseman': '5', 'shortstop': '6', 'left fielder': '7', 'center fielder': '8', 'right fielder': '9'
    }

     # ✅ Extract fielder names in order from the sentence
    fielders_found = re.findall(r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)", description)

    # ✅ Convert fielders to their corresponding numbers
    involved_fielders = [fielders[f] for f in fielders_found if f in fielders]

    # ✅ Ensure groundout abbreviation follows correct order
    if len(involved_fielders) >= 2:
        return f"GO{'-'.join(involved_fielders)}"
    elif len(involved_fielders) == 1:
        return f"GO{involved_fielders[0]}U"  # Unassisted Groundout
    
    return "GO"

def parse_reached_on_error(description):
    fielders = {
        'pitcher': '1', 'catcher': '2', 'first baseman': '3', 'second baseman': '4', 
        'third baseman': '5', 'shortstop': '6', 'left fielder': '7', 'center fielder': '8', 'right fielder': '9'
    }

    # ✅ Determine error type
    if 'throwing error' in description or 'errant throw' in description or 'bad throw' in description:
        error_type = '<sup>t</sup>' #Superscript 't' for throwing error
    elif 'fielding error' in description or 'misplay' in description or 'dropped ball' in description:
        error_type = '<sup>f</sup>' #Superscript 'f' for fielding error
    else:
        error_type = ''  # Default: No extra notation

    # ✅ Extract fielder names in order from the sentence
    fielders_found = re.findall(r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)", description)

    # ✅ Convert fielders to their corresponding numbers
    involved_fielders = [fielders[f] for f in fielders_found if f in fielders]

    # ✅ Ensure the abbreviation follows the correct order
    if len(involved_fielders) >= 2:
        return f"E{error_type}{'-'.join(involved_fielders)}"  # Multiple fielders involved
    elif len(involved_fielders) == 1:
        return f"E{error_type}{involved_fielders[0]}"  # Single fielder error

    return "E"  # Generic error if fielder is not identified

# matcher = Matcher(nlp.vocab)

# # **Define NLP Patterns for Play Outcomes**
# patterns = [
#     [{"LOWER": "grounds out"}, {"LOWER": "to"}, {"LOWER": {"REGEX": "[1-9]b"}}],  # "Groundout to 6B"
#     [{"LOWER": "flyball"}, {"LOWER": "to"}, {"LOWER": {"REGEX": "[1-9]b"}}],  # "Flyball to 7B"
# ]

# # **Add Patterns to the Matcher**
# for i, pattern in enumerate(patterns):
#     matcher.add(f"PLAY_PATTERN_{i}", [pattern])

# def parse_play_with_nlp(events, description):

#     if not isinstance(events, str) or events.strip() == "":
#         return '-'

#     if not isinstance(description, str) or description.strip() == "":
#         return '-'
    
#      # ✅ Initialize outcome by first calling the rule-based function
#     outcome = parse_play_description(events, description)

#     if outcome != '-':  # ✅ If rule-based parsing succeeds, return immediately
#         return outcome

#     doc = nlp(description.lower())

#     matches = matcher(doc)
#     for match_id, start, end, in matches:
#         span = doc[start:end]
#         text = span.text.lower()

#         if "home run" in text:
#             return "HR"
#         if "struck out looking" in text:
#             return "Ʞ"  # Backward K
#         if "struck out swinging" in text:
#             return "K"
#         if "groundout to" in text:
#             return f"GO{text[-1]}"
#         if "flyball to" in text:
#             return f"F{text[-1]}"

#     # **Fallback to Rule-Based Parsing**
#     return outcome

# Generate a miniature baseball diamond graphic
class BaseballDiamondGraphic(Flowable):
    def __init__(self, outcome, size=20, bases=None):
        """
        outcome: The result of the play.
        size: The size of the diamond graphic.
        bases: List of bases occupied [1, 2, 3, 'home'].
        """
        Flowable.__init__(self)
        self.outcome = outcome
        self.size = size
        self.bases = bases if bases else []  # Default to no runners
        
##        player_entities = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']    
##
##        # Example: If player names are found, we can extract them for further analysis
##        if player_entities:
##            print("Player(s) involved:", player_entities)             ##i thought this could be helpful to identify what players go where on the bases but idk how or where to put it.
        
    def draw(self):
        d = self.canv
        size = self.size
        center = size / 2
        offset = size / 8  # Offset for base circles

        # Translate the drawing origin to center of the cell
        d.translate(center, center)

        # Draw the diamond symmetrically around the center
        d.setStrokeColor(colors.black)
        d.setLineWidth(1)
        d.line(0, -center, center, 0)       # First base
        d.line(center, 0, 0, center)        # Second base
        d.line(0, center, -center, 0)       # Third base
        d.line(-center, 0, 0, -center)      # Home plate

        # Highlight bases with filled circles
        d.setFillColor(colors.red)  # Base runner color
        base_positions = {
            1: (center / 2, 0),        # First base
            2: (0, center / 2),        # Second base
            3: (-center / 2, 0),       # Third base
            'home': (0, -center / 2)   # Home plate
        }
        for base in self.bases:
            if base in base_positions:
                x, y = base_positions[base]
                d.circle(x, y, offset, fill=1)

        # Draw outcome text at center
        d.setFont("Helvetica-Bold", size / 4)
        d.setFillColor(colors.black)
        d.drawCentredString(0, -size / 8, self.outcome)
# Generate the PDF
def save_combined_scorecard(df, output_pdf):
    pdfmetrics.registerFont(TTFont('Unifont', "C:/Users/ncflo/.matplotlib/Unifont.ttf"))
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []
    outcome_style = ParagraphStyle(
    'OutcomeStyle',
    parent=styles['Normal'],
    fontName='Unifont',  # ✅ Force Unifont in play descriptions
    fontSize=10
)

    # Title
    elements.append(Paragraph("<b>Game 5 World Series Play-By-Play</b>", styles['Title']))
    elements.append(Spacer(1, 20))


    for idx, (team, batter_stats) in enumerate(team_scorecards.items()):
        print(f"DEBUG: Generating scorecard for {team}")
        if batter_stats is None or batter_stats.empty:
            print(f"WARNING: No data for team {team}")
            continue
        # Add a team title
        elements.append(Paragraph(f"<b>Team: {team}</b>", styles['Heading2']))
        elements.append(Spacer(1, 10))
        

        # Table headers
        table_data = [['batter'] + [str(i) for i in range(1, 10)] + ['PA', 'H', 'BB', 'SO']]

        # Add rows with diamond graphics
        for _, row in batter_stats.iterrows():
            row_data = [row['batter']]
            for i in range(1, 10):
                outcome = row[str(i)]
                row_data.append(Paragraph(outcome if outcome else '-', outcome_style, encoding='utf-8'))
            row_data += [row['PA'], row['H'], row['BB'], row['SO']]
            table_data.append(row_data)

    # Define table
        col_widths = [100] + [50] * 9 + [60] * 4
        table = Table(table_data, colWidths=col_widths, rowHeights=50, repeatRows=1)
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1,-1), 'Unifont'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),  # ✅ Adjust Font Size if Needed
    ]))

        elements.append(table)

        # **Pitcher Stats Section**
        elements.append(Paragraph(f"<b>Pitching Stats - {team}</b>", styles['Heading2']))
        elements.append(Spacer(1, 10))

        pitcher_table_data = [['pitcher', 'IP', 'ER', 'H', 'HR', 'BB', 'K']]
        for pitcher, stats in pitcher_stats.items():
            team_pitchers = play_by_play_data.loc[play_by_play_data['pitcher'] == pitcher, ['home_team', 'away_team']].drop_duplicates()
            print(f"DEBUG: Checking pitcher -> {pitcher}, Stats: {stats}")  # ✅ Debugging Pitchers
            if team in team_pitchers:  # ✅ Ensure correct team is displayed
                print(f"DEBUG: Adding pitcher stats for {pitcher} to {team}")
                pitcher_table_data.append([
                    pitcher, round(stats['IP'], 1), stats['ER'], stats['H'], stats['HR'], stats['BB'], stats['K']
                ])

        # **Format Pitcher Table**
        pitcher_table = Table(pitcher_table_data, colWidths=[120, 50, 50, 50, 50, 50, 50])
        pitcher_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ]))

        elements.append(pitcher_table)
        elements.append(Spacer(1, 20))

        if idx < len(team_scorecards) - 1:
            elements.append(PageBreak()) # ✅ Separate each team's scorecard onto a new page
        
    #Generate the PDF
    doc.build(elements)
    print(f"PDF scorecard saved at: {output_pdf}")

# Process play-by-play data
def process_play_by_play(df):
    
    # ✅ Extract relevant columns
    df_filtered = df[['inning', 'inning_topbot', 'des', 'batter']].copy()

    # ✅ Keep only rows where 'events' is not null (final play results)
    df_filtered = df.dropna(subset=['events']).copy()  # ✅ Keeps only plays with final outcomes

    # ✅ Convert inning to T1, B1 format
    df_filtered['Refined Inn'] = df_filtered.apply(
        lambda row: f"T{row['inning']}" if row['inning_topbot'] == "Top" else f"B{row['inning']}", axis=1
    )

    df_filtered['team'] = df_filtered.apply(
    lambda row: row['away_team'] if row['inning_topbot'] == "Top" else row['home_team'], axis=1
    )

     # ✅ Parse play descriptions
    df_filtered['Outcome'] = df_filtered.apply(lambda row: parse_play_description(row['events'], row['des']), axis=1)

    # ✅ Ensure batter column is numeric
    df['batter'] = df['batter'].astype(int)

    # ✅ Get unique batter IDs
    batter_ids = df['batter'].dropna().unique()

    # ✅ Fetch player names from MLBAM IDs
    player_info = playerid_reverse_lookup(batter_ids, key_type='mlbam')

    # ✅ Create a dictionary mapping ID -> Full Name
    id_to_name = {row['key_mlbam']: f"{row['name_first']} {row['name_last']}" for _, row in player_info.iterrows()}

    # ✅ Apply name extraction using the player ID lookup
    df_filtered['batter_name'] = df_filtered.apply(
        lambda row: extract_player_name(row['des'], row['batter'], id_to_name), axis=1
    )

    # Group by team
    grouped_data = df_filtered.groupby(['team'])
    team_scorecards = {}
    pitcher_stats = {}
    play_by_play_data = df_filtered.copy()
    
    # Process each team's data separately
    for team, team_data in grouped_data:
        print(f"Processing scorecard for team: {team}")
        
        if team_data.empty:
                print(f"WARNING: No play data found for {team}!")
                continue
        
    # Initialize batter stats
    
        # ✅ Ensure 'des' column has valid batter names
        team_data['batter_name'] = team_data.apply(lambda row: extract_player_name(row['des'], row['batter'], id_to_name), axis=1)

        batting_innings = {batter: ['-'] * 9 for batter in team_data['batter_name'].dropna().unique()}
        for _, row in team_data.iterrows():
            batter = row['batter_name']
            pitcher = row['pitcher']
            inning = row['Refined Inn']
            description = row['des']

            # **Initialize Pitcher Stats If Not Tracked**
            if pitcher not in pitcher_stats:
                pitcher_stats[pitcher] = {
                    "IP": 0, "ER": 0, "K": 0, "BB": 0, "H": 0, "HR": 0
                }

            # **Track Pitching Stats**
                if 'strikeout' in description or 'struck out' in description:
                    pitcher_stats[pitcher]["K"] += 1
            
                if 'walk' in description or 'base on balls' in description:
                    pitcher_stats[pitcher]["BB"] += 1

                if 'single' in description:
                    pitcher_stats[pitcher]["H"] += 1
                elif 'double' in description:
                    pitcher_stats[pitcher]["H"] += 1
                elif 'triple' in description:
                    pitcher_stats[pitcher]["H"] += 1
                elif 'home run' in description or 'homer' in description:
                    pitcher_stats[pitcher]["H"] += 1
                    pitcher_stats[pitcher]["HR"] += 1  # ✅ Track Home Runs Allowed
            
                if 'scores' in description:
                    pitcher_stats[pitcher]["ER"] += 1

                if any(word in description for word in ['groundout', 'flyout', 'popout', 'lineout', 'strikeout']):
                    pitcher_stats[pitcher]["IP"] += 1/3  # ✅ Each out = 1/3 of an inning

                # **DEBUG PRINT: Confirm Pitcher Stats Are Being Updated**
                print(f"DEBUG: Pitcher Stats Updated -> {pitcher}: {pitcher_stats[pitcher]}")
            
            if inning and inning[1:].isdigit():
                inning_index = int(inning[1:]) - 1
                outcome = parse_play_description(row['events'],row['des'])
                    
                # **DEBUG PRINT: Ensure Play Outcome is Processed**
                print(f"DEBUG: Storing -> {batter}, Inning {inning_index+1}, Outcome: {outcome}")
                
                # ✅ If the batter already has an at-bat in this inning, append outcome with "/"
                if batting_innings[batter][inning_index] != '-':
                    batting_innings[batter][inning_index] += f"/{outcome}"
                else:
                    batting_innings[batter][inning_index] = outcome
                
            
    # Create batter stats DataFrame
        batter_stats = pd.DataFrame([{"batter": batter, **{str(i + 1): val for i, val in enumerate(innings)}}
            for batter, innings in batting_innings.items()
    ])

    # Add statistics columns
        batter_stats["PA"] = batter_stats[[str(i) for i in range(1, 10)]].apply(lambda row: sum(val != '-' for val in row), axis=1)
        batter_stats["H"] = batter_stats[[str(i) for i in range(1, 10)]].apply(lambda row: sum(val in ['1B', '2B', '3B', 'HR'] for val in row), axis=1)
        batter_stats["BB"] = batter_stats[[str(i) for i in range(1, 10)]].apply(lambda row: sum(val == 'BB' for val in row), axis=1)
        batter_stats["SO"] = batter_stats[[str(i) for i in range(1, 10)]].apply(lambda row: sum(val == 'K' for val in row), axis=1)

        # **Initialize Pitcher Stats If Not Tracked**
        if pitcher not in pitcher_stats:
            pitcher_stats[pitcher] = {
                    "IP": 0, "ER": 0, "K": 0, "BB": 0, "H": 0, "HR": 0
                }

            # **Track Pitching Stats**
            if 'strikeout' in description or 'struck out' in description:
                pitcher_stats[pitcher]["K"] += 1
            
            if 'walk' in description or 'base on balls' in description:
                pitcher_stats[pitcher]["BB"] += 1

            if 'single' in description:
                pitcher_stats[pitcher]["H"] += 1
            elif 'double' in description:
                pitcher_stats[pitcher]["H"] += 1
            elif 'triple' in description:
                pitcher_stats[pitcher]["H"] += 1
            elif 'home run' in description or 'homer' in description:
                pitcher_stats[pitcher]["H"] += 1
                pitcher_stats[pitcher]["HR"] += 1  # ✅ Track Home Runs Allowed
            
            if 'scores' in description:
                pitcher_stats[pitcher]["ER"] += 1

            if any(word in description for word in ['groundout', 'flyout', 'popout', 'lineout', 'strikeout']):
                pitcher_stats[pitcher]["IP"] += 1/3  # ✅ Each out = 1/3 of an inning

            # **DEBUG PRINT: Confirm Pitcher Stats Are Being Updated**
            print(f"DEBUG: Pitcher Stats Updated -> {pitcher}: {pitcher_stats[pitcher]}")
    
    # Store team stats separately
        print(f"DEBUG: Batter stats for {team}: \n{batter_stats}")
        print(batter_stats)

        if not batter_stats.empty:
            team_scorecards[team] = batter_stats
        else: print(f"WARNING: No valid batting stats for {team}, skipping storage.")
    
    print(f"DEBUG: Final teams stored: {list(team_scorecards.keys())}")  # ✅ Check if LAD appears
    
    return team_scorecards, pitcher_stats, play_by_play_data

# Main Execution
if __name__ == "__main__":
    game_date = input("Enter the game date (YYYY-MM-DD): ").strip()
    home_team = input("Enter the home team abbreviation (e.g., NYY): ").strip().upper()
    away_team = input("Enter the away team abbreviation (e.g., LAD): ").strip().upper()
    output_pdf = input("Enter the path to save the PDF scorecard: ").strip()

    df = fetch_statcast_data(game_date, home_team, away_team)

    if df is not None:
        team_scorecards, pitcher_stats, play_by_play_data = process_play_by_play(df)
        save_combined_scorecard(play_by_play_data, output_pdf)  # ✅ Pass play_by_play_data
            
 # Keep the window open if running in Windows (for debugging purposes)
    input("Press Enter to exit...")
    