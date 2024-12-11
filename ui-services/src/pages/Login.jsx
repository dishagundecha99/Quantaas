
import GoogleIcon from "@mui/icons-material/Google";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Typography from "@mui/material/Typography";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import classes from "../components/Header.module.css";
import axiosConfig from "../utils/AxiosConfig";
const cardStyle = {
    display: "block",
    minWidth: "30%",
    boxShadow: "0px 4px 5px -2px rgba(0,0,0,0.2), 0px 7px 10px 1px rgba(0,0,0,0.14), 0px 2px 16px 1px rgba(0,0,0,0.12)",
};

const getAuthGoogle = async (navigate) => {
    console.log("getAuthGoogle");
    const response = await axiosConfig.get("/auth", {params: {"user-id": "adch9983@colorado.edu"}});
    console.log("response", response.data);
    localStorage.setItem("logged_in", response.data.logged_in);
    localStorage.setItem("user_id", response.data.user_id);
    navigate("/dashboard");
};

const Login = (props) => {

    const navigate = useNavigate();

    return (
        <>
            <Header />
            <div align="center" className={classes["overlay"]}>
                <Card style={cardStyle} variant="outlined">
                    <CardMedia
                        sx={{ height: 200, paddingtop: "56.25%", marginTop: "10px", objectFit: "cover" }}
                    >
                        <img src={"/logo.png"} alt="logo" style={{ height: "100%", width: "100%", objectFit: "cover" }} />
                    </CardMedia>
                    <CardContent sx={{ marginTop: "10px" }}>
                        <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                            Please sign in with Google
                        </Typography>
                    </CardContent>
                    <CardActions style={{ justifyContent: "center", marginBottom: "10px" }}>
                        <Button
                            variant="outlined"
                            sx={{ boxShadow: 7 }}
                            startIcon={<GoogleIcon />}
                            size="large"
                            onClick={() => getAuthGoogle(navigate)}
                        >
                            Sign in
                        </Button>
                    </CardActions>
                </Card>
            </div>
        </>
    );
};

export default Login;