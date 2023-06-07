function varargout = SoundRecorderDemo2(varargin)
% SOUNDRECORDERDEMO MATLAB code for SoundRecorderDemo.fig
%      SOUNDRECORDERDEMO, by itself, creates a new SOUNDRECORDERDEMO or raises the existing
%      singleton*.
%
%      H = SOUNDRECORDERDEMO returns the handle to a new SOUNDRECORDERDEMO or the handle to
%      the existing singleton*.
%
%      SOUNDRECORDERDEMO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SOUNDRECORDERDEMO.M with the given input arguments.
%
%      SOUNDRECORDERDEMO('Property','Value',...) creates a new SOUNDRECORDERDEMO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SoundRecorderDemo_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SoundRecorderDemo_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SoundRecorderDemo

% Last Modified by GUIDE v2.5 06-Jun-2023 16:35:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SoundRecorderDemo_OpeningFcn, ...
                   'gui_OutputFcn',  @SoundRecorderDemo_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SoundRecorderDemo is made visible.
function SoundRecorderDemo_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SoundRecorderDemo (see VARARGIN)

% Choose default command line output for SoundRecorderDemo
handles.output = hObject;
handles.recObj1 = audiorecorder(8000, 16, 1,1);
handles.recObj1.TimerFcn={@RecDisplay,handles};
handles.recObj1.TimerPeriod=0.25;
handles.playSpeed=1;
% Update handles structure

guidata(hObject, handles);

% UIWAIT makes SoundRecorderDemo wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SoundRecorderDemo_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

record(handles.recObj1);
pause(30);
stop(handles.recObj1)
handles.myRecording1 = getaudiodata(handles.recObj1);
guidata(hObject, handles);
pushbutton4_Callback(hObject, eventdata, handles)

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
stop(handles.recObj1)
handles.myRecording1 = getaudiodata(handles.recObj1);
guidata(hObject, handles);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.myRecording1 = getaudiodata(handles.recObj1);
handles.playObj1 = audioplayer(handles.myRecording1,handles.playSpeed*handles.recObj1.SampleRate);
play(handles.playObj1);
guidata(hObject, handles);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% [file,path] = uiputfile(['soundDemo_Speed' num2str(handles.playSpeed) '.mat'],'Save recorded sound');
[file,path] = uiputfile(['E:\Users\Lenovo\Desktop\噪声\100\USERS015_.mat'],'Save recorded sound');
EarPCG = handles.myRecording1;
fs = handles.recObj1.SampleRate;
if file
%     audiowrite([path '\' file],handles.myRecording,handles.playSpeed*handles.recObj.SampleRate)
    save([path '\' file],'GT_PCG', 'EarPCG', 'fs')
end

function RecDisplay(hObject, eventdata,handles)
%handles
handles.myRecording1 = getaudiodata(handles.recObj1);
% axes(handles.axes1)
if (length(handles.myRecording1)<10*handles.recObj1.SampleRate)
    plot(handles.axes1,(1:length(handles.myRecording1))/handles.recObj1.SampleRate,handles.myRecording1)
else
    plot(handles.axes1,(1:10*handles.recObj1.SampleRate)/handles.recObj1.SampleRate,handles.myRecording1(end-10*handles.recObj1.SampleRate+1:end))
end
drawnow;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
handles.playSpeed=str2double(get(hObject,'String'));
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
path = uigetdir(['K:\test'], "选择数据根目录");
preprocess(path)
% plot([1 10],[1 10])




% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path] = uiputfile(['K:\Dataset\All\Dataset_cwt_f26_05_330\Total_IMG\testset\*.png'],'选择识别样本');
img = imread([path '\' file]);
imshow(img)
handles.path1 = [path '\' file];
guidata(hObject, handles);





% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
obj = py.importlib.import_module('test_for_one_01');
py.importlib.reload(obj);
out = py.test_for_one_01.predict(handles.path1);
set(handles.edit3,'string', string(out))

% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path] = uiputfile(['K:\Dataset\All\Dataset_cwt_f26_05_330\renzheng_test\1\testset\*.png'],'选择认证样本');
img = imread([path '\' file]);
imshow(img)
handles.path2 = [path '\' file];
guidata(hObject, handles);



% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
obj = py.importlib.import_module('density');
py.importlib.reload(obj);
% obj = py.importlib.import_module('utils_for_class');
% py.importlib.reload(obj);
obj = py.importlib.import_module('test_for_one_02');
py.importlib.reload(obj);
out = py.test_for_one_02.predict(handles.path2);
set(handles.edit4,'string', string(out))



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
