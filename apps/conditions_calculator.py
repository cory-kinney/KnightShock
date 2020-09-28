import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app

mode = 'initial-conditions'
thermal_equilibrium = True

T1 = 298
P1 = ""
T2 = ""
P2 = ""
T4 = 298
P4 = ""
T5 = ""
P5 = ""

driven_mixture = ""
driver_mixture = ""

area_ratio = 1

M = ""
u = ""

n_clicks = 0


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup([dbc.Label("P4 (bar)"),
                                   dbc.Input(id='p4-input', value=P4, disabled=(mode != 'initial-conditions'))])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("P1 (bar)"),
                                   dbc.Input(id='p1-input', value=P1, disabled=(mode != 'initial-conditions'))])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("P5 (bar)"),
                                   dbc.Input(id='p5-input', value=P5, disabled=(mode != 'target-conditions'))])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("P2 (bar)"),
                                   dbc.Input(id='p2-input', value=P2, disabled=True)])
                ])
            ], form=True),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup([dbc.Label("T4 (K)"),
                                   dbc.Input(id='t4-input', value=T4, disabled=thermal_equilibrium)])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("T1 (K)"),
                                   dbc.Input(id='t1-input', value=T1)])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("T5 (K)"),
                                   dbc.Input(id='t5-input', value=T5, disabled=(mode != 'target-conditions'))])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("T2 (K)"),
                                   dbc.Input(id='t2-input', value=T2, disabled=True)])
                ])
            ], form=True),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup([dbc.Label("Driver mixture"),
                                   dbc.Textarea(id='driver-mixture-input', value=driver_mixture)])
                ]),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("Driven mixture"),
                                   dbc.Textarea(id='driven-mixture-input', value=driven_mixture)])
                ])
            ], form=True),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup([dbc.Label("Area Ratio"),
                                   dbc.Input(id='area-ratio-input', value=area_ratio)])
                ], width=3),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("Mach number"),
                                   dbc.Input(id='incident-mach-input', value=M, disabled=True)])
                ], width=3),
                dbc.Col([
                    dbc.FormGroup([dbc.Label("Shock velocity (m/s)"),
                                   dbc.Input(id='shock-velocity-input', value=u, disabled=True)])
                ], width=3)
            ], form=True),
            dbc.Button("Calculate", id='calculate-button', color='primary', disabled=False)
        ], width=8),
        dbc.Col([
            dbc.Label("Mode"),
            dbc.RadioItems(
                id='mode-radios',
                options=[
                    {'label': "Initial Conditions", 'value': 'initial-conditions'},
                    {'label': "Target Conditions", 'value': 'target-conditions'}
                ],
                value=mode
            ),
            html.Br(),
            dbc.Label("Assumptions"),
            dbc.FormGroup([
                dbc.Checkbox(id='thermal-equilibrium-checkbox', className='form-check-input',
                             checked=thermal_equilibrium),
                dbc.Label("Thermal equilibrium", className='form-check-label'),
            ], check=True)
        ])
    ]),
])


@app.callback([Output('p4-input', 'disabled'), Output('p1-input', 'disabled'),
               Output('p5-input', 'disabled'), Output('t5-input', 'disabled')],
              [Input('mode-radios', 'value')])
def update_mode(mode_input):
    global mode, P4, P1, P5, T5
    mode = mode_input
    if mode_input == 'initial-conditions':
        # TODO: Change data
        return False, False, True, True
    elif mode_input == 'target-conditions':
        # TODO: Change data
        return True, True, False, False


@app.callback(Output('t4-input', 'disabled'),
              [Input('thermal-equilibrium-checkbox', 'checked')])
def update_thermal_equilibrium(thermal_equilibrium_checked):
    global thermal_equilibrium, T1, T4
    if thermal_equilibrium_checked:
        thermal_equilibrium = True
        return True
    else:
        thermal_equilibrium = False
        return False


@app.callback([Output('t4-input', 'value'), Output('t1-input', 'invalid')],
              [Input('t1-input', 'value'), Input('t5-input', 'value'),
               Input('thermal-equilibrium-checkbox', 'checked')])
def update_T1(T1_input, T5_input, thermal_equilibrium_checked):
    global T1, T4
    try:
        T1 = float(T1_input)
        if T1 > 0:
            if thermal_equilibrium_checked:
                T4 = T1
            try:
                if T1 >= float(T5_input):
                    return T4, True
            except ValueError:
                pass
            return T4, False
        else:
            return T4, True
    except ValueError:
        if T1_input == "":
            if thermal_equilibrium_checked:
                T4 = ""
            return T4, False
        return T4, True


@app.callback(Output('t4-input', 'invalid'),
              [Input('t4-input', 'value')])
def update_T4(T4_input):
    global T4
    try:
        T4 = float(T4_input)
        if T4 > 0:
            return False
        else:
            return True
    except ValueError:
        if T4_input == "":
            return False
        return True


@app.callback(Output('t5-input', 'invalid'),
              [Input('t5-input', 'value'), Input('t1-input', 'value')])
def update_T5(T5_input, T1_input):
    global T5
    try:
        T5 = float(T5_input)
        if T5 > 0:
            try:
                if T5 <= float(T1_input):
                    return True
            except ValueError:
                pass
            return False
        else:
            return True
    except ValueError:
        if T5_input == "":
            return False
        return True


@app.callback(Output('p1-input', 'invalid'),
              [Input('p1-input', 'value'), Input('p4-input', 'value')])
def update_P1(P1_input, P4_input):
    global P1
    try:
        P1 = float(P1_input)
        if P1 > 0:
            try:
                if P1 >= float(P4_input):
                    return True
            except ValueError:
                pass
            return False
        else:
            return True
    except ValueError:
        if P1_input == "":
            return False
        return True


@app.callback(Output('p4-input', 'invalid'),
              [Input('p4-input', 'value'), Input('p1-input', 'value')])
def update_P4(P4_input, P1_input):
    global P4
    try:
        P4 = float(P4_input)
        if P4 > 0:
            try:
                if P4 <= float(P1_input):
                    return True
            except ValueError:
                pass
            return False
        else:
            return True
    except ValueError:
        if P4_input == "":
            return False
        return True


@app.callback(Output('p5-input', 'invalid'),
              [Input('p5-input', 'value')])
def update_P5(P5_input):
    global P5
    try:
        P5 = float(P5_input)
        if P5 <= 0:
            return True
        else:
            return False
    except ValueError:
        if P5_input == "":
            return False
        return True


@app.callback(Output('area-ratio-input', 'invalid'),
              [Input('area-ratio-input', 'value')])
def update_area_ratio(area_ratio_input):
    global area_ratio
    try:
        area_ratio = float(area_ratio_input)
        if area_ratio < 1:
            return True
        else:
            return False
    except ValueError:
        if area_ratio_input == "":
            return False
        return True


@app.callback(Output('calculate-button', 'disabled'),
              [Input('p1-input', 'value'), Input('t1-input', 'value'), Input('p4-input', 'value'),
               Input('t4-input', 'value'), Input('p5-input', 'value'), Input('t5-input', 'value'),
               Input('driver-mixture-input', 'value'), Input('driven-mixture-input', 'value'),
               Input('area-ratio-input', 'value')])
def update_calculate_button_state(P1_input, T1_input, P4_input, T4_input, P5_input, T5_input,
                                  driver_mixture_input, driven_mixture_input, area_ratio_input):
    try:
        if float(T1_input) <= 0 or float(T4_input) <= 0 or float(area_ratio_input) < 1:
            return True
        if mode == 'initial-conditions' and (float(P1_input) <= 0 or float(P4_input) <= float(P1_input)):
            return True
        if mode == 'target-conditions' and (float(P5_input) <= 0 or float(T5_input) <= float(T1_input)):
            return True
        # Check mixture input

        return False

    except TypeError:
        return True

    except ValueError:
        return True


@app.callback([Output('p1-input', 'value'), Output('p4-input', 'value'),
               Output('p5-input', 'value'), Output('t5-input', 'value'),
               Output('p2-input', 'value'), Output('t2-input', 'value'),
               Output('incident-mach-input', 'value'), Output('shock-velocity-input', 'value')],
              [Input('calculate-button', 'n_clicks')])
def calculate(calculate_n_clicks):
    global n_clicks

    if calculate_n_clicks is not None:
        if calculate_n_clicks > n_clicks:
            pass

        n_clicks = calculate_n_clicks

    return P1, P4, P5, T5, P2, T2, M, u
