"""Cuisine taxonomy: map 700+ Google categories and 85 DOHMH cuisine labels to 25 groups.

Rules are ordered regular expressions on the lowercased label; the first match wins, so specific
cuisines come before broad ones (e.g. "Korean barbecue" is Korean, "Japanese steakhouse" is Japanese).
Groups are of two kinds:
  cuisine  a food tradition the user can steer toward (Chinese, Mexican, Vegetarian & healthy, ...)
  format   a way of eating that says little about cuisine (cafe, fast food, bar, breakfast, deli,
           venue: hotels, event spaces, food halls, markets and caterers with no cuisine category)
Generic labels ("Restaurant", "Delivery Restaurant", "Family restaurant") match nothing.
"""
import re

GROUPS = {
    # key: (display name, kind)
    "chinese": ("Chinese", "cuisine"),
    "korean": ("Korean", "cuisine"),
    "japanese": ("Japanese", "cuisine"),
    "southeast_asian": ("Southeast Asian", "cuisine"),
    "south_asian": ("South Asian", "cuisine"),
    "asian_other": ("Asian, other", "cuisine"),
    "pizza": ("Pizza", "cuisine"),
    "italian": ("Italian", "cuisine"),
    "mexican": ("Mexican", "cuisine"),
    "caribbean": ("Caribbean", "cuisine"),
    "latin_american": ("Latin American", "cuisine"),
    "middle_eastern": ("Middle Eastern & Mediterranean", "cuisine"),
    "african": ("African", "cuisine"),
    "european": ("European", "cuisine"),
    "jewish_kosher": ("Jewish & kosher", "cuisine"),
    "seafood": ("Seafood", "cuisine"),
    "vegetarian_healthy": ("Vegetarian & healthy", "cuisine"),
    "american": ("American", "cuisine"),
    "breakfast_brunch": ("Breakfast & brunch", "format"),
    "cafe_dessert": ("Cafe, bakery & dessert", "format"),
    "sandwich_deli": ("Sandwich & deli", "format"),
    "fast_food": ("Fast food & chicken", "format"),
    "bar_pub": ("Bar & pub", "format"),
    "venue": ("Venue, market & catering", "format"),
    "general": ("General", "format"),
}

RULES = [
    (r"snack bar", "fast_food"),
    (r"mixed buffet", "sandwich_deli"),
    (r"irish pub|sports bar|hookah bar", None),                    # resolved by later format rules
    (r"hong kong|cantonese|sichuan|szechuan|hunan|shanghainese|mandarin|dim sum|dumpling|chinese|"
     r"taiwanese|hot pot|\bwok\b|porridge|congee", "chinese"),
    (r"korean", "korean"),
    (r"japanese|sushi|ramen|izakaya|udon|soba|teppanyaki|yakitori|shabu|tempura|tonkatsu|okonomiyaki|"
     r"kaiseki|omakase", "japanese"),
    (r"thai|vietnamese|\bpho\b|malaysian|indonesian|filipino|burmese|singaporean|southeast asian|"
     r"cambodian|laotian", "southeast_asian"),
    (r"indian|pakistani|bangladeshi|nepal|sri lankan|himalayan|tibetan|south asian", "south_asian"),
    (r"asian|noodle|mongolian|hawaiian|poke", "asian_other"),
    (r"pizza", "pizza"),
    (r"italian|pasta|tuscan|roman restaurant|neapolitan|sicilian", "italian"),
    (r"tex-mex|mexican|taco|burrito|torta|taqueria|southwestern", "mexican"),
    (r"jamaican|haitian|trinidad|guyanese|west indian|caribbean", "caribbean"),
    (r"latin|peruvian|colombian|dominican|salvadoran|ecuadorian|argentin|brazilian|venezuelan|"
     r"puerto rican|cuban|honduran|guatemalan|uruguayan|chilean|bolivian|nicaraguan|south american|"
     r"empanada", "latin_american"),
    (r"mediterranean|greek|middle eastern|lebanese|turkish|israeli|falafel|gyro|kebab|persian|iranian|"
     r"afghan|uzbek|moroccan|egyptian|yemen|syrian|armenian|halal", "middle_eastern"),
    (r"ethiopian|african|nigerian|senegalese|ghanaian", "african"),
    (r"french|cr[eê]perie|brasserie|bistro|spanish|tapas|basque|portuguese|german|austrian|belgian|"
     r"polish|russian|ukrainian|georgian|european|scandinavian|british|irish|english|czech|continental|"
     r"swiss|hungarian|australian", "european"),
    (r"jewish|kosher", "jewish_kosher"),
    (r"seafood|oyster|crab|fish", "seafood"),
    (r"vegetarian|vegan|health food|salad|raw food|organic restaurant|gluten-free", "vegetarian_healthy"),
    (r"american|hamburger|burger|diner|southern|soul food|cajun|creole|barbecue|bbq|steak|chophouse|"
     r"hot ?dog|cheesesteak|hoagie|chicken wings|californian", "american"),
    (r"breakfast|brunch|pancake|waffle", "breakfast_brunch"),
    (r"coffee|cafe|espresso|bakery|bagel|donut|dessert|ice cream|pastry|cake|patisserie|frozen yogurt|"
     r"bubble tea|tea house|juice|smoothie|chocolate|cookie|creamery|pretzel", "cafe_dessert"),
    (r"sandwich|\bdeli\b|soup", "sandwich_deli"),
    (r"fast food|chicken|fried", "fast_food"),
    (r"\bbar\b|\bpub\b|lounge|gastropub|brewpub|beer|wine bar|cocktail|\bgrill\b", "bar_pub"),
    (r"hotel|event venue|wedding venue|banquet hall|night ?club|live music|concert hall|bowling|food court|"
     r"food hall|market|grocery|supermarket|caterer|catering|event planner|dinner theater", "venue"),
]
_COMPILED = [(re.compile(p), g) for p, g in RULES]


def group_of(label):
    """Group key for one category or DOHMH label, or None for generic labels."""
    s = str(label).lower()
    skip_specific = False
    for rx, g in _COMPILED:
        if rx.search(s):
            if g is None:
                skip_specific = True
                continue
            if skip_specific and GROUPS[g][1] == "cuisine":
                continue
            return g
    return None


def assign(categories, dohmh=None):
    """Primary group, all groups, and where the primary came from, for one restaurant.

    Google lists a business's primary category first, so the first category that maps to a
    cuisine wins; then the DOHMH cuisine; then the first format group other than "venue" (so a
    Subway listed as "Caterer; Fast food restaurant" is fast food); then "venue"; then "general".
    """
    cats = [c for c in str(categories or "").split("; ") if c]
    g_cats = [group_of(c) for c in cats]
    g_doh = group_of(dohmh) if dohmh and dohmh not in ("Other", "Not Listed/Not Applicable") else None
    all_groups = sorted({g for g in g_cats + [g_doh] if g})
    for g in g_cats:
        if g and GROUPS[g][1] == "cuisine":
            return g, all_groups, "google"
    if g_doh and GROUPS[g_doh][1] == "cuisine":
        return g_doh, all_groups, "dohmh"
    for g in g_cats:
        if g and g != "venue":
            return g, all_groups, "google"
    if g_doh and g_doh != "venue":
        return g_doh, all_groups, "dohmh"
    if "venue" in g_cats:
        return "venue", all_groups, "google"
    return "general", all_groups, "none"
