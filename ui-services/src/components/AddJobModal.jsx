import AddTaskIcon from '@mui/icons-material/AddTask';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { Box, DialogContent, FormControl, FormLabel, Grid } from '@mui/material';
import AppBar from '@mui/material/AppBar';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ListItem from '@mui/material/ListItem';
import Slide from '@mui/material/Slide';
import TextField from '@mui/material/TextField';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import React, { useEffect, useState } from 'react';
import Select, { components } from 'react-select';
import { useForm } from 'react-hook-form';
import axios from 'axios';

const Transition = React.forwardRef(function Transition(props, ref) {
    return <Slide direction="up" ref={ref} {...props} />;
});

function AddJobModal(props) {
    const { open, handleClose } = props;
    const { register, handleSubmit } = useForm();

    const [exp, setExp] = useState('');
    const [task, setTask] = useState('');
    const [model, setModel] = useState('');
    const [lr, setLr] = useState('');
    const [wd, setWd] = useState('');
    const [ws, setWs] = useState('');
    const [bs, setBs] = useState('');
    const [ep, setEp] = useState('');

    
    useEffect(() => {
        console.log("Set Options : ", task);
    }, [task, model, lr, wd, ws, bs, ep]);

    // TODO: Rename as per backend mapping
    const task_options = [
        { label: "Sentiment Classification", value: "classification" },
        { label: "Question Answering", value: "Question Answering" }, { label: "Summarization", value: "Summarization" },
        { label: "Machine Tranlation", value: "Machine Translation" }
    ]
    const model_options = {
        "classification": [
            { label: "Distil Roberta", value: "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" },
            { label: "Finbert", value: "finbert" },
            { label: "FinTwitBERT", value: "FinTwitBERT-sentiment" },
            { label: "Sentinet-v1", value: "sentinet-v1" }
        ],
        "Question Answering": [
            { label: "Distil-Bert-uncased", value: "Falconsai/question_answering_v2" },
            { label: "Atom-7B", value: "Atom-7B" },
            { label: "Tiny Roberta", value: "tiny-roberta-squadv2" },
            { label: "Ferret-3B", value: "ferret-3b" }
        ],
        "Summarization": [
            { label: "Bill Sum Model", value: "stevhliu/my_awesome_billsum_model" },
            { label: "Bart", value: "bart-large-cnn" },
            { label: "DistillBart", value: "distillbart-cnn" },
            { label: "MBart", value: "mbart-v2" }
        ],
        "Machine Translation": [
            { label: "T5 small", value: "t5-small" },
            { label: "Neural Machine Translation", value: "neural-machine-translation" },
            { label: "TransformerNMT", value: "TransformerNMT" },
            { label: "PosNMT", value: "pos-nmt-v1" }
        ]
    }
    const DropdownIndicator = (props) => {
        return (
            <components.DropdownIndicator {...props}>
                <SearchIcon />
            </components.DropdownIndicator>
        );
    };

    const user_id = localStorage.getItem("user_id");

    const onSubmit = (data) => {
        const formData = new FormData();
        let hyp = {
            "learning_rate": lr.split(','),
            "weight_decay": wd.split(','),
            "warmup_steps": ws.split(','),
            "max_epochs": ep.split(','),
            "batch_size": bs.split(',')
        }
        formData.append("exp_name", exp);
        formData.append("task_type", task);
        formData.append("model_name", model);
        formData.append("hyperparams", JSON.stringify(hyp));
        formData.append("train_user_file", data["train_user_file"][0]);
        formData.append("test_user_file", data["test_user_file"][0]);
        console.log(formData);
        
        axios
            .request({
                url: window._env_.API_URL+"/submit-job", method: "POST", headers: {
                    'content-type': 'multipart/form-data'
                }, params: { "user-id": user_id }, data: formData })
            .then((response) => {
                console.log("Created", response);
            }).catch((error) => {
                console.log("Error Failed", error);
            });
        handleClose();
    };


    return (
        <React.Fragment>
            <Dialog
                fullScreen
                open={open}
                onClose={handleClose}
                TransitionComponent={Transition}
            >

                <AppBar sx={{ position: 'relative', bgcolor: "rgb(120, 9, 81)" }}>
                    <Toolbar>
                        <IconButton
                            edge="start"
                            color="inherit"
                            onClick={handleClose}
                            aria-label="close"
                        >
                            <CloseIcon />
                        </IconButton>
                        <Typography sx={{ ml: 2, flex: 1 }} variant="h6" component="div">
                            Job Details
                        </Typography>
                        <Button autoFocus variant='contained' color='primary' onClick={handleSubmit(onSubmit)}>
                            <AddTaskIcon sx={{ pr: 0.5 }} fontSize='medium' />
                            Add Job
                        </Button>
                    </Toolbar>
                </AppBar>
                <DialogContent>
                    <FormControl>
                        <FormLabel maxWidth="md" sx={{ fontSize: 24, pb: 1 }}> Experiment : </FormLabel>
                        <Grid container rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }} sx={{ pb: 3 }} alignItems='center' justifyContent="flex-end">
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Experimanet Name : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>
                                        <TextField
                                            fullWidth
                                            label="Enter Experiment Name"
                                            variant="outlined"
                                            size="medium"
                                            name="exp_name"
                                            onChange={(e) => {
                                                console.log('Exp Name : ', e.target.value)
                                                setExp(e.target.value)
                                            }}
                                            
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Task Type : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>
                                        <Select
                                            id="task_type"
                                            type="search"
                                            label="Task Type"
                                            width="100%"
                                            components={{ DropdownIndicator }}
                                            placeholder={"Select Task Type"}
                                            styles={{
                                                control: (baseStyles) => ({
                                                    ...baseStyles,
                                                    fontSize: 15,
                                                    height: 55
                                                }),
                                                menu: (baseStyles) => ({
                                                    ...baseStyles,
                                                    fontSize: 15,
                                                }),
                                            }}                                           
                                            onChange={(e) => {
                                                console.log('Task Type :', e.value)
                                                setTask(e.value)
                                                // register["task_type"] = e.value
                                            }}
                                            
                                            options={task_options}
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Model Name : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>
                                        <Select
                                            id="model_name"
                                            type="search"
                                            label="Model Name"
                                            components={{ DropdownIndicator }}
                                            placeholder={"Model Name"}
                                            styles={{
                                                control: (baseStyles) => ({
                                                    ...baseStyles,
                                                    fontSize: 15,
                                                    height: 55
                                                }),
                                                menu: (baseStyles) => ({
                                                    ...baseStyles,
                                                    fontSize: 15,
                                                }),
                                            }}
                                            width="100%"
                                            onChange={(e) => {
                                                console.log('Model Name : ', e)
                                                // register["model_name"] = e.value
                                                setModel(e.value)
                                            }}
                                            
                                            options={model_options[task]}
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                        </Grid>
                        <Divider />
                        <FormLabel maxWidth="md" sx={{ fontSize: 24, pt: 1 }}> Datasets : </FormLabel>
                        <Grid container rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }} sx={{ pt: 3, pb: 3 }} alignItems='center' justifyContent="flex-end">
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Upload Train Dataset : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Button
                                        component="label"
                                        variant="outlined"
                                        startIcon={<UploadFileIcon />}
                                        sx={{ marginRight: "1rem" }}
                                    >
                                        Upload Training Data
                                        <input type="file" accept=".csv" hidden onChange={(e) => {
                                            console.log('Train file Name : ', e.target.files)
                                            // register("train_user_file", { value: e.target.files[0] });
                                        }} {...register("train_user_file")} />
                                    </Button>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Upload Test Dataset : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Button
                                        component="label"
                                        variant="outlined"
                                        startIcon={<UploadFileIcon />}
                                        sx={{ marginRight: "1rem" }}
                                    >
                                        Upload Test Data
                                        <input type="file" accept=".csv" hidden onChange = {(e) => {
                                            console.log('Test file Name : ', e.target.files)
                                            // register("test_user_file", { value: e.target.files[0] });
                                        }} {...register("test_user_file")}/>
                                    </Button>
                                </ListItem>
                            </Grid>
                        </Grid>
                        <Divider />
                        <FormLabel maxWidth="md" sx={{ fontSize: 24, pt: 1 }}> Hyper Parameters : </FormLabel>
                        <Grid container rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }} sx={{ pt: 3, pb: 3 }} alignItems='center' justifyContent="flex-end">
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Learning Rate : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>

                                        <TextField
                                            label="Enter Learning Rate"
                                            variant="outlined"
                                            size="medium"
                                            name="learning_rate"
                                            onChange={(e) => {
                                                console.log('Learning Rate : ', e.target.value)
                                                // register("learning_rate", { value: e.target.value });
                                                setLr(e.target.value)
                                            }}
                                            // {...register("learning_rate")}
                                            fullWidth
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Weight Decay : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>

                                        <TextField
                                            label="Enter Weight Decay"
                                            variant="outlined"
                                            size="medium"
                                            name="weight_decay"
                                            onChange={(e) => {
                                                console.log('Weight Decay : ', e.target.value)
                                                // register("weight_decay", { value: e.target.value });
                                                setWd(e.target.value)
                                            }}
                                            // {...register("weight_decay")}
                                            fullWidth
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Batch Size : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>

                                        <TextField
                                            label="Enter Batch Size"
                                            variant="outlined"
                                            size="medium"
                                            name="batch_size"
                                            onChange={(e) => {
                                                console.log('Batch Size : ', e.target.value)
                                                // register("batch_size", { value: e.target.value });
                                                setBs(e.target.value)
                                            }}
                                            // {...register("batch_size")}
                                            fullWidth
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Warmup Steps : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>

                                        <TextField
                                            label="Enter Warmup Steps"
                                            variant="outlined"
                                            size="medium"
                                            name="warmup_steps"
                                            onChange={(e) => {
                                                console.log('Warmup Steps : ', e.target.value)
                                                // register("warmup_steps", { value: e.target.value });
                                                setWs(e.target.value)
                                            }}
                                            // {...register("warmup_steps")}
                                            fullWidth
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                <Box textAlign='right'>
                                    <FormLabel maxWidth="md" sx={{ p: 0, ml: 0, fontSize: 16 }}> Max Epochs : </FormLabel>
                                </Box>
                            </Grid>
                            <Grid xs={6} alignItems='center'>
                                {/* <ListItem >
                                    <FormLabel maxWidth="md" > Max Epochs : </FormLabel>
                                </ListItem> */}
                                <ListItem>
                                    <Container maxWidth="md" sx={{ p: 0, ml: 0 }} disableGutters>

                                        <TextField
                                            label="Enter Max Epochs"
                                            variant="outlined"
                                            size="medium"
                                            name="max_epochs"
                                            onChange={(e) => {
                                                console.log('Max Epochs : ', e.target.value)
                                                // register("max_epochs", { value: e.target.value });
                                                setEp(e.target.value)
                                            }}
                                            // {...register("max_epochs")}
                                            fullWidth
                                            margin="normal"
                                        />
                                    </Container>
                                </ListItem>
                            </Grid>
                        </Grid>
                    </FormControl>
                </DialogContent>

            </Dialog>
        </React.Fragment>
    );
}

export default AddJobModal;