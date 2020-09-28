import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import conditions_calculator


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        brand="KNIGHT SHOCK",
        color='dark',
        dark=True
    ),
    html.Br(),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    return conditions_calculator.layout


if __name__ == "__main__":
    app.run_server(debug=True)
